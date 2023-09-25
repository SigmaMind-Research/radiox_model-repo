import gradio as gr
import torch
import sentencepiece as spm
from torchvision import transforms
import io  # Import the 'io' module for bytes handling

from inference_model import RadioXModel
from load import load_model
from PIL import Image

# Load the model
model = RadioXModel.model
load_model('radiox_0.5.pt', model)

# Load the unigram model
vocab_file = 'mimic_unigram_1000.model'
dataset_vocab = spm.SentencePieceProcessor(vocab_file)

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# Define a function to preprocess the image
def preprocess_for_inference(image):
    imgs = [transform(image).unsqueeze(0)]
    cur_len = len(imgs)
    max_views = 1
    for _ in range(cur_len, max_views):
        imgs.append(torch.zeros_like(imgs[0]))
    imgs = torch.cat(imgs, dim=0)
    input_tensor = imgs.unsqueeze(1)
    return input_tensor

# Define a function to preprocess the image and generate the report
def generate_report(input_image):
    # Convert PIL.Image to bytes
    image_bytes = io.BytesIO()
    input_image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    # Convert bytes to PIL.Image
    image = Image.open(io.BytesIO(image_bytes))

    input_tensor = preprocess_for_inference(image)

    # View position
    view_pos = torch.tensor([1])

    # Prepare input_tuple
    input_tuple = (input_tensor, view_pos)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tuple = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in input_tuple)

    # Model inference
    model.eval()
    with torch.no_grad():
        output = model(input_tuple)
    # Convert token IDs to text
    candidate = ''
    predicted_ids = output[0].squeeze().tolist()
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
    return candidate

iface = gr.Interface(
    fn=generate_report,
    inputs=gr.Image(type="pil"),  # Input is a PIL.Image
    outputs="text",   # Output is text
    #capture_session=True
)

iface.launch()
