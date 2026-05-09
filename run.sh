#!/usr/bin/env bash
set -E -e -u -o pipefail

cd /home/martins3/data/leetgpu-challenges
export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LOCAL_CUDA_LIBCUDA_PATH="/lib64/libcuda.so.1"
# .venv/bin/python local_cuda/local_test.py challenges/easy/1_vector_add
easy=(
	1_vector_add
	2_matrix_multiplication
	3_matrix_transpose
	7_color_inversion
	8_matrix_addition
	9_1d_convolution
	19_reverse_array
	21_relu
	23_leaky_relu
	24_rainbow_table
	31_matrix_copy
	41_simple_inference
	52_silu
	54_swiglu
	62_value_clipping
	63_interleave
	65_geglu
	66_rgb_to_grayscale
	68_sigmoid
)

hard=(
	12_multi_head_attention
	14_multi_agent_sim
	15_sorting
	20_kmeans_clustering
	36_radix_sort
	39_Fast_Fourier_transform
	46_bfs_shortest_path
	53_casual_attention
	56_linear_attention
	59_sliding_window_attn
	73_all_pairs_shortest_paths
	74_gpt2_block
	93_llama_transformer_block
)

medium=(
	4_reduction
	5_softmax
	6_softmax_attention
	10_2d_convolution
	11_3d_convolution
	13_histogramming
	16_prefix_sum
	17_dot_product
	18_sparse_matrix_vector_multiplication
	22_gemm
	25_categorical_cross_entropy_loss
	27_mean_squared_error
	28_gaussian_blur
	29_top_k_selection
	30_batched_matrix_multiplication
	32_int8_quantized_matmul
	33_ordinary_least_squares
	34_logistic_regression
	35_monte_carlo_integration
	37_matrix_power
	38_nearest_neighbor
	40_batch_normalization
	42_2d_max_pooling
	43_count_array_element
	44_count_2d_array_element
	45_count_3d_array_element
	47_subarray_sum
	48_2d_subarray_sum
	49_3d_subarray_sum
	50_rms_normalization
	51_max_subarray_sum
	55_attn_w_linear_bias
	57_fp16_batched_matmul
	58_fp16_dot_product
	60_top_p_sampling
	61_rope_embedding
	64_weight_dequantization
	67_moe_topk_gating
	69_jacobi_stencil_2d
	70_segmented_prefix_sum
	71_parallel_merge
	72_stream_compaction
	75_sparse_matrix_dense_matrix_multiplication
	76_adder_transformer
	78_2d_fft
	80_grouped_query_attention
	81_int4_matmul
	82_linear_recurrence
	84_swiglu_mlp_block
	85_lora_linear
	87_speculative_decoding_verification
	90_causal_depthwise_conv1d
	92_decaying_causal_attention
	94_ssm_selective_scan
	96_int8_kv_cache_attention
)

for i in "${easy[@]}"; do
	.venv/bin/python local_cuda/local_test.py challenges/easy/$i
done
