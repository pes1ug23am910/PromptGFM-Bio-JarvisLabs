"""
GPU functionality test script.

Tests PyTorch CUDA availability and GPU operations.
"""

import torch

print('=' * 60)
print('GPU CONFIGURATION')
print('=' * 60)

print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'cuDNN Version: {torch.backends.cudnn.version()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'\nCurrent GPU: {torch.cuda.current_device()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    
    props = torch.cuda.get_device_properties(0)
    print(f'GPU Memory: {props.total_memory / 1024**3:.2f} GB')
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'Multi Processors: {props.multi_processor_count}')
    
    print('\n' + '=' * 60)
    print('TESTING GPU OPERATIONS')
    print('=' * 60)
    
    # Test basic tensor operations
    print('\nTest 1: Basic tensor creation and GPU transfer...')
    x = torch.randn(1000, 1000).cuda()
    print(f'✓ Created tensor on GPU: {x.device}')
    
    print('\nTest 2: Matrix multiplication on GPU...')
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f'✓ Matrix multiplication successful')
    print(f'  Result shape: {z.shape}')
    print(f'  Result device: {z.device}')
    
    print('\nTest 3: GPU memory usage...')
    print(f'  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
    print(f'  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB')
    
    print('\nTest 4: GPU computation speed test...')
    import time
    
    # CPU test
    x_cpu = torch.randn(2000, 2000)
    y_cpu = torch.randn(2000, 2000)
    start = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    
    # GPU test
    x_gpu = torch.randn(2000, 2000).cuda()
    y_gpu = torch.randn(2000, 2000).cuda()
    torch.cuda.synchronize()
    start = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f'  CPU time: {cpu_time*1000:.2f} ms')
    print(f'  GPU time: {gpu_time*1000:.2f} ms')
    print(f'  Speedup: {cpu_time/gpu_time:.2f}x')
    
    print('\n' + '=' * 60)
    print('✓ ALL GPU TESTS PASSED!')
    print('=' * 60)
    print('\n✓ Your RTX 4060 is ready for training!')
    
else:
    print('\n' + '=' * 60)
    print('✗ CUDA NOT AVAILABLE')
    print('=' * 60)
    print('\nPossible issues:')
    print('1. NVIDIA drivers not installed')
    print('2. PyTorch CPU-only version installed')
    print('3. CUDA toolkit not properly configured')
