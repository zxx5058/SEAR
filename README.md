<h1 align="center">
 Privacy-preserved Contactless Respiration Monitoring via Defocused Video with Selective Ensemble Aggregation (SEAR)
</h1>

<p align="center">
  <strong>Xinxin Zhang <sup>1</sup></sup></strong>
  .
  <strong>Feng Zheng<sup>1</sup></strong>
  .
  <strong>Guangtao Zhai<sup>2</sup></strong>
  .
  <strong>Xiao-Ping Zhang<sup>3</sup></strong>
  .
  <strong>Menghan Hu<sup>1</sup></strong>
</p>
<p align="center">

<p align="center">
  <strong><sup>1</sup>East China Normal University</strong> &nbsp;&nbsp;&nbsp;
  <strong><sup>2</sup>Shanghai Jiao Tong University</strong> &nbsp;&nbsp;&nbsp;
  <strong><sup>3</sup>Tsinghua Berkeley Shenzhen Institute</strong> &nbsp;&nbsp;&nbsp;
</p>


[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

If you have any questions, please contact Xinxin Zhang(Zhangxinxin5058@163.com) or Menghan Hu(mhhu@ce.ecnu.edu.cn).

## 🔥Blurred Dataset
  The Blurred Dataset was constructed by covering the camera lens with a black film to simulate various levels of blur. It consists of a frontal-view subset containing 80 videos across three blur levels (Blurred1, Blurred2, and Blurred3), and a lateral-view subset with 69 videos similarly divided into SBlurred1, SBlurred2, and SBlurred3. Ground-truth respiratory signals were synthetically recorded using a wireless piezoelectric respiratory belt.

➡️[SEAR Dataset Download Links](https://pan.baidu.com/s/1NLOhBYbZAv6y_WNrq_vzZw?pwd=2025)
Extraction Code: 2025

## ✨A Gentle Introduction
The SEAR (Selective Ensemble Aggregation for Respiration) algorithm integrates a Dominant Frequency Selection (DFS) module with a Phase-Constrained Sparse Reconstruction (PCSR) module. The DFS module suppresses noise by adaptively selecting dominant respiratory frequencies from multi-channel signals, while the PCSR module enhances signal fidelity by enforcing phase consistency during sparse recovery. SEAR is the first method specifically designed for respiration monitoring from blurred video inputs, enabling robust extraction of respiratory signals under degraded visual conditions. 

This is an overview of the pricacy-preserving contactless respiration monitoring system.
![image](https://github.com/zxx5058/SEAR/blob/main/ImageFolder/SEAR%20diagram.png)


