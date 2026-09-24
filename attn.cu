#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <torch/extension.h>

// T % M == 0, N % T == 0
#define M 32
#define N 32
#define H 64

// thread tile dims (elems per thread)
#define THREAD_ROWS 4
#define THREAD_COLS 2
#define NUM_THREADS 128 // (M * N) / (THREAD_ROWS * THREAD_COLS)

// choose below so that ROWS * COLS == (M * H) / NUM_THREADS
#define OUT_THREAD_ROWS 4
#define OUT_THREAD_COLS 4


__global__ void flash_attn(const float* Q_global, const float* K_global, const float* V_global, float* O_global, int seq_len) {
    // one block takes (M, H) queries and produces T / N of (M, N) patches. So one block computes a (M, T) of scores total. 
    // Later it uses the (M, T) total to compute (M, H) of out.
    __shared__ float Q_shared[M][H];
    __shared__ float K_shared[N][H];
    __shared__ float V_shared[N][H];
    __shared__ float S_shared[M][N];
    __shared__ float O_shared[M][H];

    // you need the max across the WHOLE (M, T) part.
    __shared__ float row_max_shared[M];
    __shared__ float row_sum_shared[M];

    int t_idx = threadIdx.x;
    int global_chunk_idx = blockIdx.x;

    // this block loads (M, H) (Q) to shared mem
    int q_elements_total = M * H;
    for (int i = t_idx; i < q_elements_total; i += NUM_THREADS) {
        int local_row = i / H; // where in (M, H)
        int local_col = i % H;

        int global_row = (global_chunk_idx * M) + local_row; // row in (T, H) matrix
        Q_shared[local_row][local_col] = Q_global[global_row * H + local_col];
    }

    if (t_idx < M) {
        row_max_shared[t_idx] = -INFINITY;
        row_sum_shared[t_idx] = 0.0f;
    }

    for (int i = t_idx; i < M * H; i += NUM_THREADS) {
        O_shared[i / H][i % H] = 0.0f;
    }
    __syncthreads();




    int num_n_chunks = seq_len / N; // you go along T

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


        // THIS BLOCK BELOW IS A PASTE OF compute_S_tile_in_SRAM kernel in notes.txt. 
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

            float row_max_new = fmaxf(row_max_shared[row], max_local);
            float exp_diff = expf(row_max_shared[row] - row_max_new); // e^(old - new) to mul row_sum_old by

            for (int d = 0; d < H; d++) {
                O_shared[row][d] *= exp_diff;
            }

            // this loop calculates local sum of exp with new max and writes to S
            float local_row_sum = 0.0f;
            for (int i = 0; i < N; i++) {
                float p = expf(S_shared[row][i] - row_max_new);

                S_shared[row][i] = p;
                local_row_sum += p;
            }

            // you correct the sum for calculating next tiles on the run (mul old by exp diff)
            float row_sum_new = row_sum_shared[row] * exp_diff + local_row_sum;

            // update shared mem so all threads can see that later
            row_max_shared[row] = row_max_new; 
            row_sum_shared[row] = row_sum_new;
        }

        // atp (M, N) patch is filled with sum of exps. Now you do matmul to turn (M, T) of S into (M, H) of OUT. V is (N, H)
        __syncthreads();

        // we have (M, N) in scores, which corresponds to "tokens m, m+1, m+2 pay this much attention to tokens n, n+1, n+2"
        // Also V of (N, H) means "tokens n, n+1, n+3 have these values"
        // so naturally you multiply (M, N) x (N, H) = (M, H) = "attention result for m, m+1, m+2 (M) for N first tokens (in V)"

        // SV = O
        int out_tiles_per_row = H / OUT_THREAD_COLS;

        // index of tile in the 2D tile grid
        int out_tile_row = t_idx / out_tiles_per_row;
        int out_tile_col = t_idx % out_tiles_per_row;

        // starting row/col
        int out_row_start = out_tile_row * OUT_THREAD_ROWS;
        int out_col_start = out_tile_col * OUT_THREAD_COLS;

        float O_local[OUT_THREAD_ROWS][OUT_THREAD_COLS] = {0.0f};

        for (int d = 0; d < N; d++) { // walk across N
            float S_reg[OUT_THREAD_ROWS];
            float V_reg[OUT_THREAD_COLS];

            for (int i = 0; i < OUT_THREAD_ROWS; i++) {
                S_reg[i] = S_shared[out_row_start + i][d];
            }

            for (int j = 0; j < OUT_THREAD_COLS; j++) {
                V_reg[j] = V_shared[d][out_col_start + j];
            }

            for (int i = 0; i < OUT_THREAD_ROWS; i++) {
                for (int j = 0; j < OUT_THREAD_COLS; j++) {
                    O_local[i][j] += S_reg[i] * V_reg[j];
                }
            }
        }

        // atp O_local has full output for a tile.

        // write result to shared
        for (int i = 0; i < OUT_THREAD_ROWS; i++) {
            for (int j = 0; j < OUT_THREAD_COLS; j++) {
                O_shared[(out_row_start + i)][out_col_start + j] += O_local[i][j];
            }
        }

        __syncthreads();
    }

    // WRITE TO VRAM!!! FINALLY (and scale)
    // you can scale now because division is distributive. You wait till you have the final sum.
    // Matmul SV multiplied ROWS of S (whole row has the same sum of row), so you can just divide ONCE at the end.

    int o_elements_total = M * H;
    for (int i = t_idx; i < o_elements_total; i += NUM_THREADS) {
        int local_row = i / H;
        int local_col = i % H;
        int global_row = (global_chunk_idx * M) + local_row;

        float prob = O_shared[local_row][local_col] / row_sum_shared[local_row];

        O_global[global_row * H + local_col] = prob;
    }
}

torch::Tensor run_flash_attn(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int seq_len = Q.size(0);

    auto O = torch::zeros_like(Q);
    int blocks = seq_len / M;
    
    flash_attn<<<blocks, NUM_THREADS>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), seq_len);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_flash_attn, "Flash attention fwd pass");
}