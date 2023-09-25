import torch 

def load_model(path, model, optimizer=None, scheduler=None):
	checkpoint = torch.load(path,map_location=torch.device('cpu'))
	# --- Model Parameters ---
	model.load_state_dict(checkpoint['model_state_dict'])
	# model.load_state_dict(checkpoint['model_state_dict'], strict=False