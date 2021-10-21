import torch
# model = torch.load("Log/finetune_on_ic17_look_up/model_0060000.pth")
# model = torch.load("Log/finetune_on_ic17_domain3/model_0050000.pth")
# model = torch.load("Log/retrival_rec10_no_da_b64_vot_veri_640/model_0065000.pth")
# model = torch.load("Log/finetune_on_ic17_rec10_no_da_b64_vot_veri_640/model_0067500.pth")
# model = torch.load("Log/finetune_on_ic17_rec10_no_da_b64_vot_veri_640/model_0075000.pth")
# model = torch.load("Log/retrival_rec10_no_da_b64_vot_veri_640/model_0085000.pth")
# model = torch.load("Log/finetune_on_ic17_rec10_no_da_b64_640/model_0067500.pth")
# model = torch.load("Log/finetune_on_ic17_based_on_6853/model_0072500.pth")
model = torch.load("Log/retrival_synthtext_chinese2/model_0082500.pth")
print(model.keys()) 
# Remove optimizer, iteration, and schedular
del model['optimizer']
del model['iteration']
del model['scheduler']
# Save the model
# torch.save(model, "./model_6489.pth")
# torch.save(model, "./model_6875.pth") #model = torch.load("Log/finetune_on_ic17_1500/model_0060000.pth")
# torch.save(model, "./model_look_up.pth") #"Log/retrival_e2e_add_retrieval_loss_10_colorjit_wordaug_b64_have_centerness_vot_veri_640/model_0090000.pth"
# torch.save(model, "./model_domain3_7078.pth")
# torch.save(model, "./model_domain3_7674.pth")
# torch.save(model, "./model_rec_ic17_7703.pth")
# torch.save(model, "./model_rec_ic17_7675.pth")
# torch.save(model, "./model_rec_synth_7018.pth")
# torch.save(model, "./model_rec_synth_ic17_7709.pth")
# torch.save(model, "./model_rec_synth_ic17_7709.pth")
# torch.save(model, "./model_rec_synth_ic17_norec_7576.pth")
torch.save(model, "./model_chinese_synth_5012.pth")