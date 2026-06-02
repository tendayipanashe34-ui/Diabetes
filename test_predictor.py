import traceback
try:
    import predictor
    print('predictor loaded')
    model, scaler = predictor.load_resources()
    print('model loaded?', model is not None)
    print('scaler loaded?', scaler is not None)
    if model is not None and scaler is not None:
        vals = predictor.default_input_list(use_gender=True)
        lbl, p = predictor.make_prediction(model, scaler, vals)
        print('prediction', lbl, p)
except Exception:
    traceback.print_exc()
