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
  The Blurred Dataset was constructed by covering the camera lens with a black film to simulate various levels of blur. It consists of a frontal-view subset containing 78 videos across three blur levels (Blurred1, Blurred2, and Blurred3), and a lateral-view subset with 62 videos similarly divided into SBlurred1, SBlurred2, and SBlurred3. Ground-truth respiratory signals were synthetically recorded using a wireless piezoelectric respiratory belt.

## ✨A Gentle Introduction
Camera-based respiration estimation raises significant privacy concerns, as captured visual data may expose identifiable personal information. Existing privacy-preserving approaches rely on software or physical blurring, causing severe spatial information loss that renders conventional respiration extraction methods ineffective, while illumination variations further complicate respiration extraction from blurred videos. To address these challenges, we propose an illumination-robust respiration extraction framework tailored for blurred videoes, and have constructed a real-world blurred dataset. The framework comprises two modules: an adaptive illumination artifact removal module that selectively suppresses illumination interference, and a multi-dimensional signal quality assessment module that enables adaptive ROI selection and reliable respiration extraction. To the best of our knowledge, this is the first study to develop an adaptive ROI-based respiration extraction method specifically for privacy-preserving blurred videos. Experimental results demonstrate that the proposed method achieves state-of-the-art performance on both the self-constructed blurred dataset and the Simulated Illumination-Blurred COHFACE dataset, with mean absolute errors (MAE) of 1.34 and 1.03 breaths per minute (bpm), respectively.
This is an overview of the proposed privacy-preserving respiration monitoring framework. 
![image](https://raw.githubusercontent.com/zxx5058/SEAR/refs/heads/main/ImageFolder/Framework.png)

## ✨Experiment Result
The following shows examples of respiratory signal waveforms corresponding to the subjects in the real-scene images.
![image](https://raw.githubusercontent.com/zxx5058/SEAR/refs/heads/main/ImageFolder/Respiratory%20Signal.png)

