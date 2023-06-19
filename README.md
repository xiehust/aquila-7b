## 1 打包sagemaker镜像 
`./build_and_push.sh [region]` 

## 2. 去共享地址拉取model.tar.gz，存放到sagemaker所在的bucket中
例如 `s3://sagemaker-{region}-{accountid}/model/aquila-chat-7b/model.tar.gz` 

## 3. 运行Aquila_chat_deploy.ipynb 部署模型