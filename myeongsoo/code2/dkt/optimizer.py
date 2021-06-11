from torch.optim import Adam, AdamW

def get_optimizer(model,metric, args):
    if metric :
        param = {'params': model.parameters()}, {'params': metric.parameters()}
    else : 
        param = model.parameters()
    if args.optimizer == 'adam':
        optimizer = Adam(param, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'adamW':
        optimizer = AdamW(param, lr=args.lr, weight_decay=args.weight_decay)
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()
    
    return optimizer