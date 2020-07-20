# Depth from a single image

- python 3.7
- numpy
- pandas
- matplotlib
- torch 
- torchvision

ディレクトリ構成は以下の通りです/
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
