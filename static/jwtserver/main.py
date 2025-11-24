from flask import Flask, request, jsonify
import jwt
import datetime
from functools import wraps
import secrets

app = Flask(__name__)

# 配置信息
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['TOKEN_EXPIRY_HOURS'] = 24

# 模拟用户数据库
users_db = {
    'test@10z.com': {
        'password': '123456',
        'user_id': 'user_001',
        'name': '测试用户'
    }
}

# 存储已撤销的令牌（生产环境应使用Redis等）
revoked_tokens = set()

def generate_token(user_id, email):
    """
    生成 JWT 令牌
    """
    payload = {
        'sub': user_id,
        'email': email,
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=app.config['TOKEN_EXPIRY_HOURS']),
        'jti': secrets.token_urlsafe(16)  # 令牌唯一标识
    }
    
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token

def token_required(f):
    """
    Bearer 认证装饰器
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # 从 Authorization 头部获取令牌
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({
                'error': '令牌缺失',
                'message': '请提供有效的 Bearer 令牌'
            }), 401
        
        # 检查令牌是否已被撤销
        if token in revoked_tokens:
            return jsonify({
                'error': '令牌已撤销',
                'message': '该令牌已被撤销，请重新登录'
            }), 401
        
        try:
            # 验证 JWT 令牌
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = payload['sub']
            request.user_email = payload['email']
            request.token_jti = payload['jti']
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'error': '令牌已过期',
                'message': '令牌已过期，请重新登录'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'error': '无效令牌',
                'message': '提供的令牌无效'
            }), 401
        
        return f(*args, **kwargs)
    
    return decorated

@app.route('/')
def home():
    """
    服务首页
    """
    return jsonify({
        'message': 'Bearer 认证练习服务端',
        'endpoints': {
            'login': '/login (POST) - 获取访问令牌',
            'test': '/test (GET) - 测试受保护端点',
            'profile': '/profile (GET) - 获取用户信息',
            'logout': '/logout (POST) - 撤销令牌'
        },
        'example_user': {
            'email': 'test@10z.com',
            'password': '123456'
        }
    })

@app.route('/login', methods=['POST'])
def login():
    """
    用户登录，获取访问令牌
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': '无效请求',
                'message': '请提供 JSON 格式的登录数据'
            }), 400
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({
                'error': '参数缺失',
                'message': '请提供邮箱和密码'
            }), 400
        
        # 验证用户凭证
        user = users_db.get(email)
        if not user or user['password'] != password:
            return jsonify({
                'error': '认证失败',
                'message': '邮箱或密码错误'
            }), 401
        
        # 生成访问令牌
        access_token = generate_token(user['user_id'], email)
        
        return jsonify({
            'message': '登录成功',
            'access_token': access_token,
            'token_type': 'bearer',
            'expires_in': f"{app.config['TOKEN_EXPIRY_HOURS']}小时",
            'user': {
                'user_id': user['user_id'],
                'email': email,
                'name': user['name']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': '服务器错误',
            'message': str(e)
        }), 500

@app.route('/test', methods=['GET'])
@token_required
def test_protected_endpoint():
    """
    测试受保护端点 - 需要 Bearer 认证
    """
    return jsonify({
        'message': '恭喜！你成功访问了受保护的资源',
        'user_info': {
            'user_id': request.user_id,
            'email': request.user_email
        },
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'note': '这个端点需要有效的 Bearer 令牌才能访问'
    })

@app.route('/profile', methods=['GET'])
@token_required
def get_user_profile():
    """
    获取用户个人信息 - 需要 Bearer 认证
    """
    user = users_db.get(request.user_email)
    
    return jsonify({
        'message': '用户信息获取成功',
        'profile': {
            'user_id': user['user_id'],
            'email': request.user_email,
            'name': user['name']
        },
        'authentication': {
            'token_jti': request.token_jti,
            'authenticated': True
        }
    })

@app.route('/logout', methods=['POST'])
@token_required
def logout():
    """
    用户登出，撤销当前令牌
    """
    revoked_tokens.add(request.headers.get('Authorization').split(' ')[1])
    
    return jsonify({
        'message': '登出成功',
        'details': '当前访问令牌已被撤销'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': '端点未找到',
        'message': '请求的端点不存在'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': '方法不允许',
        'message': '该端点不支持当前请求方法'
    }), 405

if __name__ == '__main__':
    print("=" * 60)
    print("Bearer 认证练习服务端")
    print("=" * 60)
    print("默认用户:")
    print("  邮箱: test@10z.com")
    print("  密码: 123456")
    print("\n启动服务后，你可以使用以下方式测试:")
    print("1. 访问 http://localhost:5000/ 查看 API 文档")
    print("2. 发送 POST 请求到 /login 获取访问令牌")
    print("3. 使用令牌访问 /test 和 /profile 端点")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)