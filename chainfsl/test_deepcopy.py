import torch, time, copy
from src.sfl.models import SplittableResNet18
model = SplittableResNet18().cuda()
t0 = time.time()
m2 = copy.deepcopy(model.get_client_model(2)).cuda()
m3 = copy.deepcopy(model.get_server_model(2)).cuda()
print('Time:', time.time()-t0)
