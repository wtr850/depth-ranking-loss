# Depth from a single image

[Monocular Relative Depth Perception with Web Stereo Data Supervision](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_Monocular_Relative_Depth_CVPR_2018_paper.pdf)の実装です。
[論文の説明記事](https://qiita.com/wtr850/private/33144081fe496d762d26)を参照してください。 

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

 - test_imgs: テスト用画像です。それなりに画質の高いRGB画像をjpegで1０枚以上入れておいてください。
 - redweb: ReDWebの画像をここに入れます。imgsに元画像、rdsに正解の深度画像を入れてください。

### 実行
Notebook内のセルを順番に実行してください。学習終了後に.pthファイルが同ディレクトリ内に出ます。
