from fastapi import Request, Depends

def get_classifier(request: Request):
    return request.app.state.classifier

def get_extractor(request: Request):
    return request.app.state.extractor