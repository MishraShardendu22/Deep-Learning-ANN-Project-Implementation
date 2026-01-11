# Deep Learning ANN: Perceptron Implementation

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, **dependency-free C++ implementation** of a single-layer **Artificial Neural Network (Perceptron)** with the classic perceptron learning algorithm. Built for understanding neural network fundamentals and solving linearly separable classification problems.

---

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Implementation](#implementation)
- [Usage & Examples](#usage--examples)
- [API Reference](#api-reference)
- [Theory](#theory)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

- **Pure C++17** — No external dependencies
- **Minimal & readable** — ~50 lines of core code
- **Step activation** — Binary classification (0 or 1)
- **Online learning** — Updates weights sample-by-sample
- **Educational** — Perfect for learning ANN fundamentals

---

## 🚀 Quick Start

### Clone & Compile

```bash
git clone https://github.com/MishraShardendu22/Deep-Learning-ANN-Project-Implementation.git
cd Deep-Learning-ANN-Project-Implementation
g++ -std=c++17 -O2 main.cpp -o perceptron
./perceptron
```

### Expected Output

```
0 0 -> 0
0 1 -> 0
1 0 -> 0
1 1 -> 1
```

---

## 🔧 Implementation

### Complete Perceptron Class

```cpp
#include <vector>
#include <iostream>

struct Perceptron {
    std::vector<double> w;  // weight vector
    double b;               // bias
    double lr;              // learning rate

    // Constructor: initialize weights to 0, bias to 0
    Perceptron(int n_inputs, double learning_rate) 
        : w(n_inputs, 0.0), b(0.0), lr(learning_rate) {}

    // Step activation function
    int activate(double z) const {
        return z >= 0.0 ? 1 : 0;
    }

    // Predict class label for input vector x
    int predict(const std::vector<double>& x) const {
        double z = b;
        for (size_t i = 0; i < w.size(); ++i)
            z += w[i] * x[i];
        return activate(z);
    }

    // Train using perceptron learning rule
    // epochs: number of passes through the dataset
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<int>& y,
               int epochs) {
        for (int e = 0; e < epochs; ++e) {
            for (size_t i = 0; i < X.size(); ++i) {
                int y_hat = predict(X[i]);
                int error = y[i] - y_hat;
                
                // Update weights
                for (size_t j = 0; j < w.size(); ++j)
                    w[j] += lr * error * X[i][j];
                
                // Update bias
                b += lr * error;
            }
        }
    }
};
```

---

## 📖 Usage & Examples

### Example 1: AND Gate

```cpp
#include <iostream>
#include <vector>

int main() {
    // Create a perceptron with 2 inputs and learning rate 0.1
    Perceptron p(2, 0.1);

    // Training data (AND gate)
    std::vector<std::vector<double>> X = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<int> y = {0, 0, 0, 1};  // AND outputs

    // Train for 10 epochs
    p.train(X, y, 10);

    // Test predictions
    std::cout << "AND gate predictions:\n";
    for (const auto& x : X) {
        std::cout << x[0] << " AND " << x[1] 
                  << " = " << p.predict(x) << "\n";
    }

    return 0;
}
```

### Example 2: OR Gate

```cpp
// Training data (OR gate)
std::vector<std::vector<double>> X = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};
std::vector<int> y = {0, 1, 1, 1};  // OR outputs

Perceptron p(2, 0.1);
p.train(X, y, 10);
```

---

## 📚 API Reference

### Constructor

```cpp
Perceptron(int n_inputs, double learning_rate)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_inputs` | `int` | Number of input features |
| `learning_rate` | `double` | Learning rate η (typically 0.01–0.5) |

**Example:**
```cpp
Perceptron p(2, 0.1);  // 2 inputs, learning rate 0.1
```

---

### predict()

```cpp
int predict(const std::vector<double>& x) const
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `const std::vector<double>&` | Input feature vector |
| **Returns** | `int` | Class label (0 or 1) |

**Example:**
```cpp
std::vector<double> input = {1.0, 0.5};
int prediction = p.predict(input);  // Returns 0 or 1
```

---

### train()

```cpp
void train(const std::vector<std::vector<double>>& X,
           const std::vector<int>& y,
           int epochs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `const std::vector<std::vector<double>>&` | Training samples (each row is a sample) |
| `y` | `const std::vector<int>&` | Training labels (0 or 1) |
| `epochs` | `int` | Number of full passes through training data |

**Example:**
```cpp
std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
std::vector<int> y = {0, 0, 0, 1};
p.train(X, y, 10);  // Train for 10 epochs
```

---

## 🧠 Theory

### The Perceptron Model

A single-layer perceptron is a linear binary classifier that computes:

**Forward pass:**
$$z = w \cdot x + b$$
$$\hat{y} = \text{step}(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

Where:
- **w** = weight vector
- **x** = input feature vector
- **b** = bias term
- **z** = net input (weighted sum)
- **ŷ** = predicted class label

### Perceptron Learning Rule

For each training sample, compute the error and update weights:

$$\text{error} = y - \hat{y}$$

$$w \leftarrow w + \eta \cdot \text{error} \cdot x$$

$$b \leftarrow b + \eta \cdot \text{error}$$

Where **η** (eta) is the learning rate.

### Convergence Guarantee

The perceptron learning algorithm **converges in finite time** if and only if the data is **linearly separable**. For non-separable data (e.g., XOR), the algorithm will not converge.

---

## ⚠️ Limitations

| Limitation | Impact |
|-----------|--------|
| **Linear-only** | Cannot solve non-linear problems (e.g., XOR) |
| **Single layer** | No hidden layers; limited expressiveness |
| **Step activation** | Non-differentiable; difficult to extend with gradient descent |
| **No regularization** | Prone to overfitting on small datasets |
| **Binary classification** | Only 0/1 outputs; not suitable for regression |

### When NOT to use

- **Multiclass problems** → Use softmax or one-hot encoding with multiple perceptrons
- **Non-linear boundaries** → Use multi-layer networks (MLPs)
- **Continuous outputs** → Use regression models
- **Large-scale data** → Use modern ML frameworks (TensorFlow, PyTorch)

---

## 💡 Next Steps

1. **Extend to multi-class** → Use multiple perceptrons
2. **Add sigmoid activation** → Replace step with sigmoid for gradient descent
3. **Implement batch training** → Process multiple samples per update
4. **Add visualization** → Plot decision boundaries
5. **Build a multi-layer network** → Stack perceptrons with hidden layers

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Ideas for PRs

- [ ] Add unit tests (Catch2/GoogleTest)
- [ ] Create examples for OR, XOR (with limitations), NAND gates
- [ ] Add a simple visualization tool
- [ ] Implement batch/mini-batch training
- [ ] Add sigmoid or ReLU activation options

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Questions

For questions or feedback, open an issue on GitHub or reach out to the maintainers.

---

**Enjoy learning neural networks! 🚀**
- Add a separate `examples/` folder and move the AND example there ✅
- Add a simple unit test and a CI workflow (GitHub Actions) ✅
- Add badges (build, license) to the top ✅

Tell me which of the above (if any) you want next and I'll add it quickly.
