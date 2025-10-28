[# 🧠 ConvPlayground

**ConvPlayground** is a simple web app that helps you *see* how **2D convolution** works.  
You can paste a matrix and a filter, click **“Process Matrix Operations”**, and instantly see the result — or even watch the convolution happen step by step.

It’s built using **pure HTML, CSS, and JavaScript (Canvas)** — no frameworks, no setup, just open and play.

---

## 🚀 Try It Out
👉 [Open ConvPlayground](https://convplayground.onrender.com)  

---

## 🧩 Example

| 🖼️ Input Matrix | 🎛️ Filter (Kernel) | 📊 Output |
|:---------------:|:------------------:|:----------:|
| <pre>1 2 3 0<br>0 1 2 3<br>3 1 0 2<br>2 3 2 1</pre> | <pre>1 0<br>0 -1</pre> | <pre>1 2 3<br>-1 0 2<br>1 1 0</pre> |

💡 Each output value = sum of all element-wise multiplications between the filter and the matching part of the matrix.

---

## ✨ Features

- 🧮 Enter your **matrix** and **filter**
- ⚙️ Click **“Process Matrix Operations”** to get the result  
- 🎞️ Watch the **animated steps** of the convolution  
- 🧠 100% client-side — no installs, no dependencies

---

## ⚙️ How to Use

1. Clone or download this repo  
2. run `app.py` in your browser  
3. Paste your matrix and kernel  
4. Click **“Process Matrix Operations”** to see the result 🎬  

That’s it — easy, fast, and fun.

---

## 📜 License
MIT © [Alliche Amine Mohammed]  
Made with ❤️ to make convolutions easy to understand.

---

⭐ If you enjoy it, don’t forget to give it a star!
