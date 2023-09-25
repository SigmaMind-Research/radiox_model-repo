import torch 
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torchvision import transforms
from PIL import Image
import io
from radiox_model import RadioXModel
from ts.torch_handler.base_handler import BaseHandler

# Initialize your RadioXModel here using the load_model function
	# --- Model Parameters ---
model = RadioXModel.model

# Initialize SentencePiece
vocab_file = 'mimic_unigram_1000.model'
dataset_vocab = spm.SentencePieceProcessor(vocab_file)

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

def preprocess_for_inference(image):
    imgs = [transform(image).unsqueeze(0)]
    cur_len = len(imgs)
    max_views = 1
    for _ in range(cur_len, max_views):
        imgs.append(torch.zeros_like(imgs[0]))
    imgs = torch.cat(imgs, dim=0)
    input_tensor = imgs.unsqueeze(1)
    return input_tensor

class CustomHandler(BaseHandler):
    def initialize(self, context):
        self.model_path = 'radiox_0.5.pt'
        self.model = model
        self.context = context
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def preprocess(self, data):
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")       
        #image_bytes = io.BytesIO()
        # image.save(image_bytes, format='PNG')
        # image_bytes = image_bytes.getvalue()
        # Convert bytes to PIL.Image        
        image = Image.open(io.BytesIO(image)).convert('RGB')
        input_tensor = preprocess_for_inference(image)
        view_pos = torch.tensor([1])

        # Prepare input_tuple
        input_tuple = (input_tensor, view_pos)
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tuple = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in input_tuple)

        return input_tuple

    def inference(self, input_data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
        return output

    def postprocess(self, inference_output):
        # Convert token IDs to text
        candidate = ''
        predicted_ids = inference_output[0].squeeze().tolist()
        for j in range(len(predicted_ids)):
            tok = dataset_vocab.id_to_piece(predicted_ids[j])
            if tok == '</s>':
                break
            elif tok == '<s>':
                continue
            elif tok == '▁':
                if len(candidate) and candidate[-1] != ' ':
                    candidate += ' '
            elif tok in [',', '.', '-', ':']:
                if len(candidate) and candidate[-1] != ' ':
                    candidate += ' ' + tok + ' ' 
                else:
                    candidate += tok + ' '
            else:
                candidate += tok
        result = []
        return result.append(candidate)


    def handle(self, data, context):
        input_data = self.preprocess(data)
        model_output = self.inference(input_data)
        api_response = self.postprocess(model_output)
        return api_response

