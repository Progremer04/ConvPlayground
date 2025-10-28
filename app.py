# matrix_visualizer.py
from flask import Flask, render_template_string, request, jsonify
import numpy as np
import os

app = Flask(__name__)

def parse_matrix(matrix_data):
    """Parse matrix from grid input"""
    try:
        rows = len(matrix_data)
        cols = len(matrix_data[0]) if rows > 0 else 0
        
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                try:
                    value = float(matrix_data[i][j])
                    row.append(value)
                except (ValueError, IndexError):
                    row.append(0.0)
            matrix.append(row)
        return np.array(matrix)
    except Exception as e:
        raise ValueError(f"Invalid matrix format: {str(e)}")

def apply_convolution(matrix, kernel, stride=1, padding=0):
    """Apply convolution operation"""
    if padding > 0:
        matrix = np.pad(matrix, padding, mode='constant')
    
    output_height = (matrix.shape[0] - kernel.shape[0]) // stride + 1
    output_width = (matrix.shape[1] - kernel.shape[1]) // stride + 1
    
    result = np.zeros((output_height, output_width))
    
    for i in range(0, output_height):
        for j in range(0, output_width):
            region = matrix[i*stride:i*stride+kernel.shape[0], 
                           j*stride:j*stride+kernel.shape[1]]
            result[i, j] = np.sum(region * kernel)
    
    return result

def apply_pooling(matrix, pool_size=2, stride=2, mode='max'):
    """Apply pooling operation"""
    if pool_size == 0:  # Pooling disabled
        return matrix
        
    output_height = (matrix.shape[0] - pool_size) // stride + 1
    output_width = (matrix.shape[1] - pool_size) // stride + 1
    
    result = np.zeros((output_height, output_width))
    
    for i in range(0, output_height):
        for j in range(0, output_width):
            region = matrix[i*stride:i*stride+pool_size, 
                           j*stride:j*stride+pool_size]
            
            if mode == 'max':
                result[i, j] = np.max(region)
            elif mode == 'avg':
                result[i, j] = np.mean(region)
            elif mode == 'min':
                result[i, j] = np.min(region)
    
    return result

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Matrix Visualizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    animation: {
                        'float': 'float 6s ease-in-out infinite',
                        'glow': 'glow 2s ease-in-out infinite alternate',
                        'matrix': 'matrix 0.5s ease-in-out',
                        'slideIn': 'slideIn 0.5s ease-out',
                        'bounceIn': 'bounceIn 0.6s ease-out',
                    },
                    keyframes: {
                        float: {
                            '0%, 100%': { transform: 'translateY(0)' },
                            '50%': { transform: 'translateY(-10px)' }
                        },
                        glow: {
                            '0%': { boxShadow: '0 0 5px #3b82f6, 0 0 10px #3b82f6, 0 0 15px #3b82f6' },
                            '100%': { boxShadow: '0 0 10px #3b82f6, 0 0 20px #3b82f6, 0 0 30px #3b82f6' }
                        },
                        matrix: {
                            '0%': { opacity: '0', transform: 'scale(0.8) rotateX(-90deg)' },
                            '100%': { opacity: '1', transform: 'scale(1) rotateX(0)' }
                        },
                        slideIn: {
                            '0%': { opacity: '0', transform: 'translateX(-20px)' },
                            '100%': { opacity: '1', transform: 'translateX(0)' }
                        },
                        bounceIn: {
                            '0%': { opacity: '0', transform: 'scale(0.3)' },
                            '50%': { opacity: '1', transform: 'scale(1.05)' },
                            '70%': { transform: 'scale(0.9)' },
                            '100%': { opacity: '1', transform: 'scale(1)' }
                        }
                    }
                }
            }
        }
    </script>
    <style>
        .matrix-grid {
            display: grid;
            gap: 4px;
        }
        .matrix-cell {
            transition: all 0.3s ease;
            border: 1px solid #374151;
        }
        .matrix-cell:hover {
            transform: scale(1.1);
            z-index: 10;
        }
        .matrix-cell input {
            background: transparent;
            border: none;
            width: 100%;
            text-align: center;
            color: white;
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .cyber-border {
            border: 1px solid #3b82f6;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
        }
        .digital-text {
            color: #00ff00;
            text-shadow: 0 0 5px #00ff00;
        }
        .gradient-bg {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        }
        .neon-glow {
            box-shadow: 0 0 10px #3b82f6, 0 0 20px #3b82f6, 0 0 30px #3b82f6;
        }
        .operation-step {
            border-left: 3px solid #3b82f6;
            padding-left: 1rem;
            margin-left: 0.5rem;
        }
    </style>
</head>
<body class="gradient-bg text-white min-h-screen">
    <!-- Animated Background -->
    <div class="fixed inset-0 opacity-10">
        <div class="absolute inset-0 bg-gradient-to-br from-blue-900 to-purple-900"></div>
        <div class="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxkZWZzPjxwYXR0ZXJuIGlkPSJwYXR0ZXJuIiB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHBhdHRlcm5Vbml0cz0idXNlclNwYWNlT25Vc2UiIHBhdHRlcm5UcmFuc2Zvcm09InJvdGF0ZSg0NSkiPjxyZWN0IHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjA1KSIvPjwvcGF0dGVybj48L2RlZnM+PHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0idXJsKCNwYXR0ZXJuKSIvPjwvc3ZnPg==')]"></div>
    </div>

    <div class="relative z-10 container mx-auto px-4 py-4 md:py-8">
        <!-- Header -->
        <header class="text-center mb-8 md:mb-12 fade-in">
            <div class="inline-block animate-float">
                <h1 class="text-2xl md:text-4xl lg:text-6xl font-bold digital-text mb-2 md:mb-4">NEURAL MATRIX VISUALIZER</h1>
            </div>
            <p class="text-sm md:text-lg text-gray-300 max-w-2xl mx-auto px-2">Advanced visualization of convolution and pooling operations with real-time matrix processing</p>
        </header>

        <div class="grid grid-cols-1 xl:grid-cols-2 gap-4 md:gap-8">
            <!-- Input Section -->
            <div class="bg-gray-800 rounded-xl md:rounded-2xl cyber-border p-4 md:p-6 fade-in">
                <h2 class="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6 flex items-center">
                    <i class="fas fa-edit mr-2 md:mr-3 text-cyan-400"></i> Matrix Configuration
                </h2>
                
                <form id="matrixForm" class="space-y-4 md:space-y-6">
                    <!-- Matrix Dimensions -->
                    <div class="grid grid-cols-2 gap-2 md:gap-4">
                        <div>
                            <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Matrix Rows</label>
                            <input 
                                type="number" 
                                id="matrixRows" 
                                min="1" 
                                max="8"
                                value="4"
                                class="w-full p-2 md:p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all text-sm md:text-base"
                            >
                        </div>
                        <div>
                            <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Matrix Columns</label>
                            <input 
                                type="number" 
                                id="matrixCols" 
                                min="1" 
                                max="8"
                                value="4"
                                class="w-full p-2 md:p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all text-sm md:text-base"
                            >
                        </div>
                    </div>
                    
                    <!-- Input Matrix Grid -->
                    <div>
                        <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Input Matrix</label>
                        <div id="matrixGrid" class="p-2 md:p-4 bg-gray-900 rounded-lg cyber-border max-w-full overflow-auto">
                            <!-- Dynamic grid will be generated here -->
                        </div>
                    </div>
                    
                    <!-- Kernel Dimensions -->
                    <div class="grid grid-cols-2 gap-2 md:gap-4">
                        <div>
                            <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Kernel Rows</label>
                            <input 
                                type="number" 
                                id="kernelRows" 
                                min="1" 
                                max="5"
                                value="3"
                                class="w-full p-2 md:p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all text-sm md:text-base"
                            >
                        </div>
                        <div>
                            <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Kernel Columns</label>
                            <input 
                                type="number" 
                                id="kernelCols" 
                                min="1" 
                                max="5"
                                value="3"
                                class="w-full p-2 md:p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all text-sm md:text-base"
                            >
                        </div>
                    </div>
                    
                    <!-- Kernel Matrix Grid -->
                    <div>
                        <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Convolution Kernel</label>
                        <div id="kernelGrid" class="p-2 md:p-4 bg-gray-900 rounded-lg cyber-border max-w-full overflow-auto">
                            <!-- Dynamic grid will be generated here -->
                        </div>
                    </div>
                    
                    <!-- Parameters Grid -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2 md:gap-4">
                        <!-- Stride -->
                        <div>
                            <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Stride</label>
                            <input 
                                type="number" 
                                id="strideInput" 
                                min="1" 
                                value="1"
                                class="w-full p-2 md:p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all text-sm md:text-base"
                            >
                        </div>
                        
                        <!-- Padding -->
                        <div>
                            <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Padding</label>
                            <input 
                                type="number" 
                                id="paddingInput" 
                                min="0" 
                                value="0"
                                class="w-full p-2 md:p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all text-sm md:text-base"
                            >
                        </div>
                    </div>
                    
                    <!-- Pooling Section -->
                    <div class="border-t border-gray-700 pt-2 md:pt-4">
                        <div class="flex items-center justify-between mb-2 md:mb-4">
                            <h3 class="text-base md:text-lg font-bold text-gray-300">Pooling Operations</h3>
                            <label class="inline-flex items-center cursor-pointer">
                                <input type="checkbox" id="poolingToggle" class="sr-only peer">
                                <div class="relative w-10 h-5 md:w-11 md:h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[1px] after:left-[1px] md:after:top-[2px] md:after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 md:after:h-5 md:after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
                                <span class="ml-2 text-xs md:text-sm font-medium text-gray-300">Enable Pooling</span>
                            </label>
                        </div>
                        
                        <div id="poolingParams" class="grid grid-cols-1 md:grid-cols-2 gap-2 md:gap-4 opacity-50 pointer-events-none">
                            <!-- Pool Size -->
                            <div>
                                <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Pool Size</label>
                                <input 
                                    type="number" 
                                    id="poolSizeInput" 
                                    min="1" 
                                    value="2"
                                    class="w-full p-2 md:p-3 bg-gray-700 border border-gray-600 rounded-lg text-sm md:text-base"
                                >
                            </div>
                            
                            <!-- Pool Stride -->
                            <div>
                                <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Pool Stride</label>
                                <input 
                                    type="number" 
                                    id="poolStrideInput" 
                                    min="1" 
                                    value="2"
                                    class="w-full p-2 md:p-3 bg-gray-700 border border-gray-600 rounded-lg text-sm md:text-base"
                                >
                            </div>
                            
                            <!-- Pool Mode -->
                            <div class="md:col-span-2">
                                <label class="block text-gray-300 text-sm md:text-base font-medium mb-1 md:mb-2">Pooling Mode</label>
                                <div class="flex flex-wrap gap-2 md:gap-4">
                                    <label class="inline-flex items-center text-xs md:text-sm">
                                        <input type="radio" name="poolMode" value="max" class="text-cyan-600 focus:ring-cyan-500" checked>
                                        <span class="ml-1 md:ml-2">Max Pooling</span>
                                    </label>
                                    <label class="inline-flex items-center text-xs md:text-sm">
                                        <input type="radio" name="poolMode" value="avg" class="text-cyan-600 focus:ring-cyan-500">
                                        <span class="ml-1 md:ml-2">Average Pooling</span>
                                    </label>
                                    <label class="inline-flex items-center text-xs md:text-sm">
                                        <input type="radio" name="poolMode" value="min" class="text-cyan-600 focus:ring-cyan-500">
                                        <span class="ml-1 md:ml-2">Min Pooling</span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Submit Button -->
                    <button 
                        type="submit" 
                        class="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white font-bold py-2 md:py-3 px-4 rounded-lg transition-all duration-300 transform hover:scale-105 flex items-center justify-center animate-glow text-sm md:text-base"
                    >
                        <i class="fas fa-bolt mr-2"></i> Process Matrix Operations
                    </button>
                </form>
            </div>
            
            <!-- Results Section -->
            <div class="bg-gray-800 rounded-xl md:rounded-2xl cyber-border p-4 md:p-6 fade-in">
                <h2 class="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6 flex items-center">
                    <i class="fas fa-chart-network mr-2 md:mr-3 text-cyan-400"></i> Neural Operations Result
                </h2>
                
                <div id="resultsContainer" class="space-y-4 md:space-y-8 max-h-[600px] overflow-y-auto">
                    <div class="text-center py-8 md:py-12 text-gray-400">
                        <i class="fas fa-project-diagram text-3xl md:text-5xl mb-3 md:mb-4 opacity-50"></i>
                        <p class="text-sm md:text-base">Configure your matrices and parameters, then process to see the neural operations</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Visualization Steps -->
        <div id="visualizationSteps" class="mt-4 md:mt-8 bg-gray-800 rounded-xl md:rounded-2xl cyber-border p-4 md:p-6 hidden">
            <h2 class="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6 flex items-center">
                <i class="fas fa-eye mr-2 md:mr-3 text-cyan-400"></i> Operation Visualization
            </h2>
            <div id="stepsContainer" class="space-y-4">
                <!-- Steps will be dynamically generated here -->
            </div>
        </div>
        
        <!-- Footer -->
        <footer class="mt-6 md:mt-12 text-center text-gray-400 text-xs md:text-sm fade-in">
            <p>Neural Matrix Visualizer • Advanced Convolution & Pooling Operations</p>
        </footer>
    </div>

    <script>
        // Initialize matrices
        document.addEventListener('DOMContentLoaded', function() {
            generateMatrixGrid();
            generateKernelGrid();
            
            // Set up event listeners for dimension changes
            document.getElementById('matrixRows').addEventListener('input', generateMatrixGrid);
            document.getElementById('matrixCols').addEventListener('input', generateMatrixGrid);
            document.getElementById('kernelRows').addEventListener('input', generateKernelGrid);
            document.getElementById('kernelCols').addEventListener('input', generateKernelGrid);
            
            // Pooling toggle
            document.getElementById('poolingToggle').addEventListener('change', function() {
                const poolingParams = document.getElementById('poolingParams');
                if (this.checked) {
                    poolingParams.classList.remove('opacity-50', 'pointer-events-none');
                } else {
                    poolingParams.classList.add('opacity-50', 'pointer-events-none');
                }
            });
            
            // Set default values for matrices
            setTimeout(() => {
                setDefaultMatrixValues();
                setDefaultKernelValues();
            }, 100);
        });
        
        function generateMatrixGrid() {
            const rows = parseInt(document.getElementById('matrixRows').value) || 4;
            const cols = parseInt(document.getElementById('matrixCols').value) || 4;
            const container = document.getElementById('matrixGrid');
            
            container.innerHTML = '';
            container.style.gridTemplateColumns = `repeat(${cols}, minmax(40px, 1fr))`;
            container.className = 'matrix-grid p-2 md:p-4 bg-gray-900 rounded-lg cyber-border max-w-full overflow-auto';
            
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const cell = document.createElement('div');
                    cell.className = 'matrix-cell bg-gray-800 rounded p-1 md:p-2 flex items-center justify-center min-w-[40px] min-h-[40px]';
                    
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.value = '0';
                    input.className = 'bg-transparent w-full text-center text-white outline-none text-sm md:text-base';
                    input.dataset.row = i;
                    input.dataset.col = j;
                    
                    cell.appendChild(input);
                    container.appendChild(cell);
                }
            }
        }
        
        function generateKernelGrid() {
            const rows = parseInt(document.getElementById('kernelRows').value) || 3;
            const cols = parseInt(document.getElementById('kernelCols').value) || 3;
            const container = document.getElementById('kernelGrid');
            
            container.innerHTML = '';
            container.style.gridTemplateColumns = `repeat(${cols}, minmax(40px, 1fr))`;
            container.className = 'matrix-grid p-2 md:p-4 bg-gray-900 rounded-lg cyber-border max-w-full overflow-auto';
            
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const cell = document.createElement('div');
                    cell.className = 'matrix-cell bg-gray-800 rounded p-1 md:p-2 flex items-center justify-center min-w-[40px] min-h-[40px]';
                    
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.value = '0';
                    input.className = 'bg-transparent w-full text-center text-white outline-none text-sm md:text-base';
                    input.dataset.row = i;
                    input.dataset.col = j;
                    
                    cell.appendChild(input);
                    container.appendChild(cell);
                }
            }
        }
        
        function setDefaultMatrixValues() {
            const defaultValues = [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ];
            
            const inputs = document.querySelectorAll('#matrixGrid input');
            inputs.forEach(input => {
                const row = parseInt(input.dataset.row);
                const col = parseInt(input.dataset.col);
                if (row < defaultValues.length && col < defaultValues[0].length) {
                    input.value = defaultValues[row][col];
                }
            });
        }
        
        function setDefaultKernelValues() {
            const defaultValues = [
                [1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]
            ];
            
            const inputs = document.querySelectorAll('#kernelGrid input');
            inputs.forEach(input => {
                const row = parseInt(input.dataset.row);
                const col = parseInt(input.dataset.col);
                if (row < defaultValues.length && col < defaultValues[0].length) {
                    input.value = defaultValues[row][col];
                }
            });
        }
        
        function getMatrixData(gridId) {
            const inputs = document.querySelectorAll(`#${gridId} input`);
            if (inputs.length === 0) return [];
            
            // Find dimensions
            let maxRow = 0;
            let maxCol = 0;
            inputs.forEach(input => {
                maxRow = Math.max(maxRow, parseInt(input.dataset.row));
                maxCol = Math.max(maxCol, parseInt(input.dataset.col));
            });
            
            // Create 2D array
            const rows = maxRow + 1;
            const cols = maxCol + 1;
            const matrix = Array(rows).fill().map(() => Array(cols).fill(0));
            
            // Fill with values
            inputs.forEach(input => {
                const row = parseInt(input.dataset.row);
                const col = parseInt(input.dataset.col);
                matrix[row][col] = parseFloat(input.value) || 0;
            });
            
            return matrix;
        }
        
        document.getElementById('matrixForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading state
            const resultsContainer = document.getElementById('resultsContainer');
            resultsContainer.innerHTML = `
                <div class="text-center py-8">
                    <div class="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500 mb-4"></div>
                    <p class="text-gray-400">Processing neural operations...</p>
                </div>
            `;
            
            // Hide visualization steps initially
            document.getElementById('visualizationSteps').classList.add('hidden');
            
            // Gather form data
            const formData = {
                matrix: getMatrixData('matrixGrid'),
                kernel: getMatrixData('kernelGrid'),
                stride: document.getElementById('strideInput').value,
                padding: document.getElementById('paddingInput').value,
                pool_size: document.getElementById('poolingToggle').checked ? 
                          document.getElementById('poolSizeInput').value : 0,
                pool_stride: document.getElementById('poolStrideInput').value,
                pool_mode: document.querySelector('input[name="poolMode"]:checked').value
            };
            
            // Send request to server
            fetch('/process_matrix', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultsContainer.innerHTML = `
                        <div class="bg-red-900 border border-red-700 rounded-lg p-4 text-red-200">
                            <i class="fas fa-exclamation-triangle mr-2"></i> ${data.error}
                        </div>
                    `;
                    return;
                }
                
                // Display results
                displayResults(data);
                // Show visualization steps
                document.getElementById('visualizationSteps').classList.remove('hidden');
                createVisualizationSteps(data);
            })
            .catch(error => {
                console.error('Error:', error);
                resultsContainer.innerHTML = `
                    <div class="bg-red-900 border border-red-700 rounded-lg p-4 text-red-200">
                        <i class="fas fa-exclamation-triangle mr-2"></i> Neural network connection failed.
                    </div>
                `;
            });
        });
        
        function displayResults(data) {
            const resultsContainer = document.getElementById('resultsContainer');
            let html = '';
            
            // Input Matrix
            html += `
                <div class="fade-in animate-matrix">
                    <h3 class="text-lg md:text-xl font-bold text-white mb-3 md:mb-4 flex items-center">
                        <i class="fas fa-th mr-2 text-green-400"></i> Input Matrix
                        <span class="ml-auto text-xs md:text-sm font-normal bg-green-900 text-green-300 px-2 md:px-3 py-1 rounded-full">
                            ${data.input_shape[0]} × ${data.input_shape[1]}
                        </span>
                    </h3>
                    ${renderMatrix(data.input_matrix, 'input')}
                </div>
            `;
            
            // Operations
            if (data.operations.length === 0) {
                html += `
                    <div class="text-center py-8 text-gray-400">
                        <i class="fas fa-info-circle text-2xl md:text-3xl mb-3 md:mb-4"></i>
                        <p class="text-sm md:text-base">No operations performed. Add a kernel for convolution or enable pooling.</p>
                    </div>
                `;
            }
            
            data.operations.forEach((op, index) => {
                const delay = (index + 1) * 200;
                
                if (op.type === 'convolution') {
                    html += `
                        <div class="fade-in animate-matrix" style="animation-delay: ${delay}ms">
                            <h3 class="text-lg md:text-xl font-bold text-white mb-3 md:mb-4 flex items-center">
                                <i class="fas fa-filter mr-2 text-blue-400"></i> Convolution Result
                                <span class="ml-auto text-xs md:text-sm font-normal bg-blue-900 text-blue-300 px-2 md:px-3 py-1 rounded-full">
                                    ${op.result_shape[0]} × ${op.result_shape[1]}
                                </span>
                            </h3>
                            <div class="mb-3 md:mb-4">
                                <h4 class="font-medium text-gray-300 text-sm md:text-base mb-1 md:mb-2">Convolution Kernel:</h4>
                                ${renderMatrix(op.kernel, 'kernel')}
                            </div>
                            <div class="mb-3 md:mb-4 text-xs md:text-sm text-gray-400">
                                <span class="bg-gray-700 px-2 md:px-3 py-1 rounded mr-2">Stride: ${op.stride}</span>
                                <span class="bg-gray-700 px-2 md:px-3 py-1 rounded">Padding: ${op.padding}</span>
                            </div>
                            ${renderMatrix(op.result, 'convolution')}
                        </div>
                    `;
                } else if (op.type === 'pooling') {
                    html += `
                        <div class="fade-in animate-matrix" style="animation-delay: ${delay}ms">
                            <h3 class="text-lg md:text-xl font-bold text-white mb-3 md:mb-4 flex items-center">
                                <i class="fas fa-layer-group mr-2 text-purple-400"></i> ${op.mode.charAt(0).toUpperCase() + op.mode.slice(1)} Pooling
                                <span class="ml-auto text-xs md:text-sm font-normal bg-purple-900 text-purple-300 px-2 md:px-3 py-1 rounded-full">
                                    ${op.result_shape[0]} × ${op.result_shape[1]}
                                </span>
                            </h3>
                            <div class="mb-3 md:mb-4 text-xs md:text-sm text-gray-400">
                                <span class="bg-gray-700 px-2 md:px-3 py-1 rounded mr-2">Pool Size: ${op.pool_size}</span>
                                <span class="bg-gray-700 px-2 md:px-3 py-1 rounded">Pool Stride: ${op.pool_stride}</span>
                            </div>
                            ${renderMatrix(op.result, 'pooling')}
                        </div>
                    `;
                }
            });
            
            resultsContainer.innerHTML = html;
            
            // Add animation to matrix cells after rendering
            setTimeout(() => {
                document.querySelectorAll('.matrix-cell').forEach(cell => {
                    cell.classList.add('pulse');
                });
            }, 500);
        }
        
        function createVisualizationSteps(data) {
            const stepsContainer = document.getElementById('stepsContainer');
            let html = '';
            
            // Step 1: Input Matrix
            html += `
                <div class="operation-step animate-slideIn">
                    <div class="flex items-center mb-2">
                        <div class="bg-cyan-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold mr-2">1</div>
                        <h4 class="font-bold text-white">Input Matrix</h4>
                    </div>
                    <p class="text-gray-300 text-sm mb-2">Original matrix with shape ${data.input_shape[0]} × ${data.input_shape[1]}</p>
                    ${renderMatrix(data.input_matrix, 'input')}
                </div>
            `;
            
            // Convolution Steps
            data.operations.forEach((op, index) => {
                if (op.type === 'convolution') {
                    html += `
                        <div class="operation-step animate-slideIn" style="animation-delay: ${(index+1)*100}ms">
                            <div class="flex items-center mb-2">
                                <div class="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold mr-2">${index+2}</div>
                                <h4 class="font-bold text-white">Convolution Operation</h4>
                            </div>
                            <p class="text-gray-300 text-sm mb-2">Applied kernel with stride ${op.stride} and padding ${op.padding}</p>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
                                <div>
                                    <p class="text-gray-300 text-sm mb-1">Kernel:</p>
                                    ${renderMatrix(op.kernel, 'kernel')}
                                </div>
                                <div>
                                    <p class="text-gray-300 text-sm mb-1">Result:</p>
                                    ${renderMatrix(op.result, 'convolution')}
                                </div>
                            </div>
                        </div>
                    `;
                } else if (op.type === 'pooling') {
                    html += `
                        <div class="operation-step animate-slideIn" style="animation-delay: ${(index+1)*100}ms">
                            <div class="flex items-center mb-2">
                                <div class="bg-purple-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold mr-2">${index+2}</div>
                                <h4 class="font-bold text-white">${op.mode.charAt(0).toUpperCase() + op.mode.slice(1)} Pooling</h4>
                            </div>
                            <p class="text-gray-300 text-sm mb-2">Applied ${op.mode} pooling with size ${op.pool_size} and stride ${op.pool_stride}</p>
                            ${renderMatrix(op.result, 'pooling')}
                        </div>
                    `;
                }
            });
            
            // Final Output
            const lastOp = data.operations[data.operations.length - 1];
            if (lastOp) {
                html += `
                    <div class="operation-step animate-bounceIn" style="animation-delay: ${(data.operations.length+1)*100}ms">
                        <div class="flex items-center mb-2">
                            <div class="bg-green-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold mr-2">${data.operations.length + 2}</div>
                            <h4 class="font-bold text-white">Final Output</h4>
                        </div>
                        <p class="text-gray-300 text-sm mb-2">Final matrix shape: ${lastOp.result_shape[0]} × ${lastOp.result_shape[1]}</p>
                        ${renderMatrix(lastOp.result, lastOp.type)}
                    </div>
                `;
            }
            
            stepsContainer.innerHTML = html;
        }
        
        function renderMatrix(matrix, type) {
            const rows = matrix.length;
            const cols = matrix[0] ? matrix[0].length : 0;
            
            let html = `<div class="inline-block border border-gray-700 rounded-lg overflow-auto" style="display: grid; grid-template-columns: repeat(${cols}, minmax(40px, 1fr)); gap: 2px;">`;
            
            matrix.forEach(row => {
                row.forEach(cell => {
                    const cellClass = `matrix-cell px-2 md:px-4 py-1 md:py-3 text-center font-medium transition-all duration-300 text-sm md:text-base min-w-[40px] min-h-[40px] ${getCellColor(cell, type)}`;
                    html += `<div class="${cellClass}">${typeof cell === 'number' ? cell.toFixed(2) : cell}</div>`;
                });
            });
            
            html += '</div>';
            return html;
        }
        
        function getCellColor(value, type) {
            if (type === 'input') return 'bg-green-900 hover:bg-green-800';
            if (type === 'kernel') return 'bg-blue-900 hover:bg-blue-800';
            if (type === 'convolution') return 'bg-indigo-900 hover:bg-indigo-800';
            if (type === 'pooling') return 'bg-purple-900 hover:bg-purple-800';
            return 'bg-gray-800 hover:bg-gray-700';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process_matrix', methods=['POST'])
def process_matrix():
    try:
        data = request.json
        
        # Parse input matrices
        input_matrix = parse_matrix(data['matrix'])
        
        kernel = None
        if data.get('kernel') and len(data['kernel']) > 0 and len(data['kernel'][0]) > 0:
            kernel = parse_matrix(data['kernel'])
        
        # Get parameters
        stride = int(data.get('stride', 1))
        padding = int(data.get('padding', 0))
        pool_size = int(data.get('pool_size', 0))  # Default to 0 (disabled)
        pool_stride = int(data.get('pool_stride', 2))
        pool_mode = data.get('pool_mode', 'max')
        
        results = {
            'input_matrix': input_matrix.tolist(),
            'input_shape': input_matrix.shape,
            'operations': []
        }
        
        # Apply convolution if kernel provided
        current_matrix = input_matrix
        if kernel is not None:
            conv_result = apply_convolution(input_matrix, kernel, stride, padding)
            results['operations'].append({
                'type': 'convolution',
                'kernel': kernel.tolist(),
                'stride': stride,
                'padding': padding,
                'result': conv_result.tolist(),
                'result_shape': conv_result.shape
            })
            current_matrix = conv_result
        
        # Apply pooling only if enabled
        if pool_size > 0:
            pool_result = apply_pooling(current_matrix, pool_size, pool_stride, pool_mode)
            results['operations'].append({
                'type': 'pooling',
                'pool_size': pool_size,
                'pool_stride': pool_stride,
                'mode': pool_mode,
                'result': pool_result.tolist(),
                'result_shape': pool_result.shape
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
