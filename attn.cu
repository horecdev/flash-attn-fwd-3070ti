#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

int main() {
    return 0;
}

#define M 64
#define N 32
#define H 128

// thread tile dims (elems per thread)
#define THREAD_ROWS 8
#define THREAD_COLS 2
#define NUM_THREADS 128 // (M * N) / (THREAD_ROWS * THREAD_COLS)

// __global__ void compute_S_tile_in_SRAM(
//     const float Q_shared[M][H], const float K_shared[N][H] // K is stored as N, H but we treat it as H, N (cuz transpose)
// ) {
//     int t_idx = threadIdx.x; // one thread = one tile
//     // you have a (64, 32) matrix. Thats 2048 elems
//     // each thread is responsible for a (THREAD_ROWS, THREAD_COLS) tile in the (64, 32) matrix.
//     // In total you must launch (M * N) / (THREAD_ROWS * THREAD_COLS) threads. In this case, its 128.
//     // If each thread covers 2 elems on the x-axis, then the total number of threads per row is: (N / THREAD_COLS) = 32/2 = 16
//     int tiles_per_row = N / THREAD_COLS; // how many tiles are needed per row.

//     // figure out the tile position in the (M, N) grid. 
//     int tile_row = t_idx / tiles_per_row;
//     int tile_col = t_idx % tiles_per_row;

//     // absolute position in the (64, 32) matrix
//     int row_start = tile_row * THREAD_ROWS;
//     int col_start = tile_col * THREAD_COLS;

//     // allocate personal thread's tile.
//     float S_local[THREAD_ROWS][THREAD_COLS] = {0.0f};

//     for (int d = 0; d < H; d++) {
//         float Q_reg[THREAD_ROWS]; // load 8 pieces of Q
//         float K_reg[THREAD_COLS]; // load 2 pieces of K

//         for (int i = 0; i < THREAD_ROWS; i++) {
//             // row_start cuz Q is (T, H) and row_start is the T index
//             Q_reg[i] = Q_shared[row_start + i][d];
//         }
//         for (int j = 0; j < THREAD_COLS; j++) {
//             // same here but we treat it as transposed while it is not (so the indices are inverted)
//             K_reg[j] = K_shared[col_start + j][d]; 
//         }

//         // so: you have Q_tile (M, H) and K_tile (H, N). You iterate over H (1 at a time) and grab a piece of 8 elems from Q and 2 from K along the T dim
//         // Then as you go over H you do matmul to get a piece of the (T, T)

//         for (int i = 0; i < THREAD_ROWS; i++) {
//             for (int j = 0; j < THREAD_COLS; j++) {
//                 S_local[i][j] += Q_reg[i] * K_reg[j]; // multiply everything with everything and add
//             }
//         }

//         // S_local[8][2] now has a patch of the (M, N)

//         // wrapping it up: This function takes Q_tile and K_tile, and calculates (M, N) scores matrix
//     }
// }

__global__ void flash_attn(const float* Q_global, const float* K_global, const float* V_global, int seq_len) {
    // one block takes (M, H) queries and produces T / N of (M, N) patches. So one block computes a (M, T) of scores total. 
    // Later it uses the (M, T) total to compute (M, H) of out.
    __shared__ float Q_shared[M][H];
    __shared__ float K_shared[N][H];
    __shared__ float V_shared[N][H];
    __shared__ float S_shared[M][N];
    __shared__ float final_softmax_row_sum[M];

    int t_idx = threadIdx.x;
    int m_chunk_idx = blockIdx.x;

    // this block loads (M, H) (Q) to shared mem
    int q_elements_total = M * H;
    for (int i = t_idx; i < q_elements_total; i += NUM_THREADS) {
        int local_row = i / H; // where in (M, H)
        int local_col = i % H;

        int global_row = (m_chunk_idx * M) + local_row; // row in (T, H) matrix
        Q_shared[local_row][local_col] = Q_global[global_row * H + local_col];
    }
    __syncthreads();


    int num_n_chunks = seq_len / N; // you go along T

    // you need the max across the WHOLE (M, T) part.
    float row_max_old = -INFINITY; 
    float row_sum_old = 0.0f;

    for (int n_chunk = 0; n_chunk < num_n_chunks; n_chunk++) {

        // this block loads K (N, H) to shared mem
        int k_elements_total = N * H;
        for (int i = t_idx; i < k_elements_total; i += NUM_THREADS) {
            int local_row = i / H;
            int local_col = i % H;
            
            int global_row = (n_chunk * N) + local_row;
            K_shared[local_row][local_col] = K_global[global_row * H + local_col];
        }

        int v_elements_total = N * H;
        for (int i = t_idx; i < v_elements_total; i += NUM_THREADS) {
            int local_row = i / H;
            int local_col = i % H;

            int global_row = (n_chunk * N) + local_row;
            V_shared[local_row][local_col] = V_global[global_row * H + local_col];
        }


        __syncthreads();


        // THIS BLOCK BELOW IS A PASTE OF compute_S_tile_in_SRAM kernel above. 
        // Explanations of code are above on specific example.
        int tiles_per_row = N / THREAD_COLS;

        int tile_row = t_idx / tiles_per_row;
        int tile_col = t_idx % tiles_per_row;

        int row_start = tile_row * THREAD_ROWS;
        int col_start = tile_col * THREAD_COLS;

        float S_local[THREAD_ROWS][THREAD_COLS] = {0.0f};

        for (int d = 0; d < H; d++) {
            float Q_reg[THREAD_ROWS];
            float K_reg[THREAD_COLS];

            for (int i = 0; i < THREAD_ROWS; i++) {
                Q_reg[i] = Q_shared[row_start + i][d];
            }
            for (int j = 0; j < THREAD_COLS; j++) {
                K_reg[j] = K_shared[col_start + j][d]; 
            }

            for (int i = 0; i < THREAD_ROWS; i++) {
                for (int j = 0; j < THREAD_COLS; j++) {
                    S_local[i][j] += Q_reg[i] * K_reg[j];
                }
            }
        }
        // at ths point S_local has a patch of (THREAD_ROWS, THREAD_COLS)

        float scale = 1.0f / sqrtf(static_cast<float>(H));

        for (int i = 0; i < THREAD_ROWS; i++) {
            for (int j = 0; j < THREAD_COLS; j++) {
                S_shared[row_start + i][col_start + j] = S_local[i][j] * scale;
            }
        }
        __syncthreads();

        // right now (M, N) is in S_shared

        if (t_idx < M) {
            // one thread per row looks for MAX
            int row = t_idx;

            float max_local = -INFINITY;
            for (int i = 0; i < N; i++) {
                max_local = fmaxf(max_local, S_shared[row][i]);
            }

            float row_max_new = fmaxf(row_max_old, max_local);
            float exp_diff = expf(row_max_old - row_max_new); // e^(old - new) to mul row_sum_old by

            // this loop calculates local sum of exp with new max and writes to S
            float local_row_sum = 0.0f;
            for (int i = 0; i < N; i++) {
                float p = expf(S_shared[row][i] - row_max_new);

                S_shared[row][i] = p;

                local_row_sum += p;
            }

            // you correct the sum for calculating next tiles on the run (mul old by exp diff)
            float row_sum_new = row_sum_old * exp_diff + local_row_sum;

            row_max_old = row_max_new;
            row_sum_old = row_sum_new;
        }

        // atp (M, N) paths is filled with scores. Now you do matmul to turn (M, T) of S into (M, H) of OUT. V is (N, H)
        __syncthreads();

        // we have (M, N) in scores, which corresponds to "tokens m, m+1, m+2 pay this much attention to tokens n, n+1, n+2"
        // Also V of (N, H) means "tokens n, n+1, n+3 have these values"
        // so naturally you multiply (M, N) x (N, H) = (M, H) = "attention result for m, m+1, m+2 (M) for N first tokens (in V)"
    }
}