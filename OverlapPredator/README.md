# **OverlapPredator (Modified Version)**
This repository is a modified version of **[OverlapPredator](https://github.com/overlappredator/OverlapPredator)**.

## **📌 What's Modified?**
This version includes several modifications and additional experimental scripts:
- **ROS-based real-time point cloud registration** (`ros_nodes/`)
- **Performance enhancements using Maximal Clique (MAC)** (`scripts_experimental/`)
- **Multi-cloud registration experiments**

For details on these modifications, refer to `scripts_experimental/` and `ros_nodes/`.

## **📂 Repository Structure**
OverlapPredator/ │── scripts/ # Standard PREDATOR scripts │── scripts_experimental/ # Experimental scripts using Maximal Clique │── ros_nodes/ # ROS-based real-time registration scripts │── assets/ # Visualization and results │── README.md # This document (modified) │── LICENSE # MIT License │── main.py # Main script for training and evaluation └── ...

csharp
Copy
Edit

## **📌 Original Paper & Citation**
This repository is based on OverlapPredator, originally implemented by **Shengyu Huang et al. (CVPR 2021).**

📄 **Paper**:  
🔗 [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/abs/2011.13005)  

📌 **Citation**:
```bibtex
@InProceedings{Huang_2021_CVPR,
    author    = {Huang, Shengyu and Gojcic, Zan and Usvyatsov, Mikhail and Wieser, Andreas and Schindler, Konrad},
    title     = {Predator: Registration of 3D Point Clouds With Low Overlap},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {4267-4276}
}
📌 Additional References for Maximal Clique (MAC)
bibtex
Copy
Edit
@inproceedings{zhang20233d,
  title={3D Registration with Maximal Cliques},
  author={Zhang, Xiyu and Yang, Jiaqi and Zhang, Shikun and Zhang, Yanning},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17745--17754},
  year={2023}
}

@article{yang2024mac,
  title={MAC: Maximal Cliques for 3D Registration},
  author={Yang, Jiaqi and Zhang, Xiyu and Wang, Peng and Guo, Yulan and Sun, Kun and Wu, Qiao and Zhang, Shikun and Zhang, Yanning},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
📌 License
This project follows the MIT License, as per the original OverlapPredator.

sql
Copy
Edit
MIT License

Copyright (c) 2021 Shengyu Huang, Zan Gojcic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
