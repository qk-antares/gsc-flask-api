from flask import Flask
from controller.ged_api import ged_api
from controller.gedgnn_api import gedgnn_api
from controller.tagsim_api import tagsim_api
from controller.simgnn_api import simgnn_api
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

# 注册蓝图
app.register_blueprint(ged_api)
app.register_blueprint(gedgnn_api)
app.register_blueprint(tagsim_api)
app.register_blueprint(simgnn_api)


if __name__ == '__main__':
    app.run(port=8123)
