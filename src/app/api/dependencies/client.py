from fastapi import Request, Depends

def get_config(request: Request):
    return request.app.state.config

def get_logger(request: Request):
    return request.app.state.logger

def get_classifier(request: Request):
    return request.app.state.classifier

def get_extractor(request: Request):
    return request.app.state.extractor