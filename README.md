# Deep-Learning-ANN-Project-Implementation

A small, dependency-free C++ repository that demonstrates a single-layer perceptron (binary classifier) and the classic perceptron learning rule. It's designed for education and experiments with linearly separable datasets (e.g., AND / OR).

---

## Quickstart ⚡

1. Clone the repo.
2. Compile:

```bash
g++ -std=c++17 -O2 perceptron.cpp -o perceptron
```

3. Run:

```bash
./perceptron
```

You should see predictions for the training examples (the included example trains an AND gate).

---

## Features ✅

- Minimal, readable C++17 implementation
- Single-layer perceptron with step activation
- Sample-wise (online) perceptron learning rule
- No third-party dependencies

---

## Perceptron summary (math)

- z = w · x + b
- ŷ = step(z) where step(z) = 1 if z ≥ 0 else 0

Update per sample:

- error = y − ŷ
- w ← w + η * error * x
- b ← b + η * error

> Converges only for linearly separable data (cannot learn XOR).

---

## Example (AND gate) 🔧

The repository contains a compact example that trains a perceptron to implement an AND gate. Copy this into `perceptron.cpp` if you want a single-file example.

```cpp
#include <iostream>
#include <vector>

int main() {
    Perceptron p(2, 0.1);
    std::vector<std::vector<double>> X = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<int> y = {0,0,0,1};

    p.train(X, y, 10);

    for (auto &x : X) std::cout << x[0] << " " << x[1] << " -> " << p.predict(x) << '\n';
}
```

---

## API (struct Perceptron)

- Perceptron(int n_inputs, double learning_rate)
- int predict(const std::vector<double>& x) const
- void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int epochs)

---

## Development & Tests

- This project is educational; no unit tests are included by default.
- Suggested next steps:
  - Add a small test using Catch2 or Google Test in `test/`
  - Add a GitHub Actions workflow for CI

---

## Contributing

Contributions welcome: open an issue or a PR for fixes, examples, or documentation improvements.

---

## License

MIT — see `LICENSE` (or add one) if you want to include full license text.

---

If you want, I can:
- Add a separate `examples/` folder and move the AND example there ✅
- Add a simple unit test and a CI workflow (GitHub Actions) ✅
- Add badges (build, license) to the top ✅

Tell me which of the above (if any) you want next and I'll add it quickly.
