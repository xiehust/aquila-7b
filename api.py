from fastapi import FastAPI, Request
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.model.predictor.aquila import aquila_generate
from flagai.data.tokenizer import Tokenizer
# import bminf
import uvicorn, json, datetime
import torch
import os

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE



def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI()

@app.get("/ping")
def ping():
    return {'status': 'Healthy'}

@app.post("/invocations")
async def create_item(request: Request):
    global predictor,tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    print('json_post:',json_post)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('inputs') 
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length') if json_post_list.get('max_length') else 2000
    temperature = json_post_list.get('temperature') if json_post_list.get('temperature') else 0

    from cyg_conversation import default_conversation
    conv = default_conversation.copy()

    for i,his in enumerate(history):
        conv.append_message(conv.roles[i%2], his) ##0-human,1-bot
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)

    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
    tokens = tokens[:-1]
    with torch.inference_mode():
        response = aquila_generate(tokenizer, model, [prompt], max_gen_len=max_length, top_p=0.95,
                    temperature = temperature, prompts_tokens=[tokens])
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    state_dict = "/opt/ml/model"
    model_name = 'aquilachat-7b' # 'aquila-33b'
    print('start...')
    loader = AutoLoader(
        "lm",
        model_dir=state_dict,
        model_name=model_name,
        use_cache=True)
    model = loader.get_model()
    print('model get completed')
    tokenizer = loader.get_tokenizer() 
    model.eval()
    model.half()
    model.cuda()
    print('model initialized')
    cache_dir = os.path.join(state_dict, model_name)
    predictor = Predictor(model, tokenizer)
    print('get predictor  completed')
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
