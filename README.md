## 1 打包sagemaker镜像 
`./build_and_push.sh [region]` 

## 2 下载模型文件
### （方式1）去官网下载:
https://model.baai.ac.cn/model-detail/100101

模型下载后，放入aquilachat-7b 目录下，并将这个目录打包成model.tar.gz文件  
`tar czvf model.tar.gz aquilachat-7b`  
例如：  
`aws s3 cp model.tar.gz s3://sagemaker-{region}-{accountid}/model/aquila-chat-7b/model.tar.gz`  

### （方式2）从临时S3拉取，需要提前联系开权限
### 联系AWS开通临时s3权限，直接使用已经打包好的s3://sagemaker-us-east-2-946277762357/model/aquila-chat-7b/model.tar.gz


## 3. 运行Aquila_chat_deploy.ipynb 部署模型