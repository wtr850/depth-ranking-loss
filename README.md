# Depth from a single image

### 要件
- python 3.7
- numpy
- pandas
- matplotlib
- torch 
- torchvision

### ディレクトリ構成
<pre>
├── test_imgs
│   ├── ....jpg
│   ├── ....jpg
│  
└── redweb
    ├── imgs
    │   ├── ....jpg
    │   ├── ....jpg
    │ 
    └── rds
        ├── ....png
        ├── ....png      
 </pre>

 - test_imgs: テスト用画像です。それなりに画質の高い画像を1０枚以上入れておいてください。
 - redweb: ReDWebの画像をここに入れます。imgsに元画像、rdsに正解の深度画像を入れてください。

### 実行
Notebook内のセルを順番に実行してください。学習終了後に.pthファイルが同ディレクトリ内に出ます。
