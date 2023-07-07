## Download the Data
Data can be download after signing the license `License Agreement.pdf`

Please sign the license and send it to jietingchen23@outlook.com and we will provide the link of data. To protect the copyright, the videos should be downloaded by yourself using the information we provided in the `meta.json` file.

## Folder Structure
    |--Data
    |--|--meta.json
    |--|--processed_data
    |--|--comments
    |--|--division
    |--|--scripts
    |--|--output

DIR | Explanation
--- | ---
meta.json | A dict that provides the meta information of movies. Information is obtained from [Douban Movie](https://movie.douban.com/)
processed_data  | Processed data which can be directly used for our code
comments | Live video comments obtained from [Tencent Video](https://v.qq.com/)
division  | The data split of MovieLC dataset
scipts | The movie scripts obtained from [the Website](http://assrt.net/)
output | The output folder for model running, including a checkpoint of KLVCG model on MovieLC dataset