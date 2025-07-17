import torch

def count_parameters(model_path):
    """Loads a PyTorch model and returns the total number of parameters."""
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        if 'model' in model: # Check if the loaded object is a dictionary with a 'model' key
            model = model['model']
        
        # It's possible the model is already a nn.Module or a state_dict
        if isinstance(model, torch.nn.Module):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        elif isinstance(model, dict): # Assuming it's a state_dict
            total_params = sum(p.numel() for p in model.values())
        else:
            print(f"Warning: Could not directly determine parameters for {model_path}. Model type: {type(model)}")
            return None
        return total_params
    except Exception as e:
        print(f"Error loading or processing model {model_path}: {e}")
        return None

model_with_dist_path = r"C:\Users\ben93\PycharmProjects\yolov7\runs\train\exp194\weights\best.pt"
model_without_dist_path = r"C:\Users\ben93\My Drive\Weights\flibs\flibs.pt"

params_with_dist = count_parameters(model_with_dist_path)
params_without_dist = count_parameters(model_without_dist_path)

if params_with_dist is not None:
    print(f"Model with distance ('{model_with_dist_path}') has: {params_with_dist:,} parameters.")

if params_without_dist is not None:
    print(f"Model without distance ('{model_without_dist_path}') has: {params_without_dist:,} parameters.")

if params_with_dist is not None and params_without_dist is not None:
    difference = params_with_dist - params_without_dist
    print(f"Difference in parameters: {difference:,}")
    if difference == 5385:
        print("This matches the expected difference for distance parameters (assuming nc=1 for both).")
    elif difference == 0:
        print("The number of parameters is identical. This could happen if nc was reduced by 1 when distance was added (e.g., nc=2 to nc=1+dist).")
    else:
        print("The difference does not match the simple calculation (5385). This could be due to other architectural differences or different nc values.") 