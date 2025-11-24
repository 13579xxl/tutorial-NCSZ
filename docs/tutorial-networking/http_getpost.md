---
sidebar_position: 4
---
# GET、POST 与 Bearer 认证
探索未知之境。
## HTTP 请求方法

| 方法 | 描述 | 使用场景 |
|------|------|----------|
| GET | 获取数据 | 查看网页、搜索、获取用户信息 |
| POST | 提交数据 | 登录、注册、上传文件、下单 |


## GET 请求特点

- **数据在 URL 中**：参数显示在地址栏
- **有长度限制**：URL 最大长度约 2048 字符
- **可被缓存**：浏览器会缓存 GET 请求
- **幂等操作**：多次执行结果相同
- **安全性低**：参数在地址栏可见

例如，`https://www.google.com/search?q=http&newwindow=1`就是一个GET请求，其中参数有：

| 参数名 | 值 | 
|------|------|
| `q` | http | 
| `newwindow` | 1 | 

你会发现，所有的参数都直接包括在URL中了，这无疑是对安全的不尊重。（试想一下，你在十中登录bilibili，账号密码直接**没有加密的**流出去，十中可以毫无保留的审查你的访问。


:::caution 注意
`GET`请求严禁传输机密数据！例如，`https://www.example.com/login?user=Alex&password=123`这样的请求，会直接把你的账号，密码**在网上明文传输**，绝对不要在GET请求里携带机密信息！
:::

并且，`GET`请求具有长度限制，如果你想在网站上上传图片，那么最好是使用`POST`。

### 示例

访问[南昌十中官网(http://10z.nceduc.cn)](http://10z.nceduc.cn)，GET获得一个文件

URL为：`http://10z.nceduc.cn/_sitegray/_sitegray_d.css`

参考代码为:

```python
import requests

result = requests.get('http://10z.nceduc.cn/_sitegray/_sitegray_d.css').content

```

`result`值应该为：`/*.nograyforsite{}*/`

:::tip 练习1

- 请仿照上面的示例，访问`http://10z.nceduc.cn/_sitegray/_sitegray.js`，谈一谈，内容是什么？

- 请仿照上面的示例，访问`http://10z.nceduc.cn/style/33.png`，谈一谈，这张图片是什么？

- 请你进一步将这张图片转换为`base64`格式

> 你可以使用`base64`库的`base64.b64encode`函数。

:::


## POST 请求特点

- **数据在请求体中**：参数不在地址栏显示
- **无长度限制**：可上传大文件
- **不被缓存**：浏览器通常不缓存 POST
- **安全性较高**：参数在请求体中

例如,张三制作了一个上传图片的API,相较于`GET`请求，POST则可以通过下列代码上传
```python
import requests

requests.post('http://a.b.com/upload', data={
    'image': '图片的base64'
})
```

:::tip 练习2

- 请你携带**练习1第3题**的数据，请求`https://httpbin.org/post`，内容为`{'image': '图片base64'}`，它返回了什么？

- 思考一下，为什么要把图片放在`body`中传输？
:::

## Bearer 认证介绍

### 什么是 Bearer 认证？

Bearer 认证是一种基于令牌（Token）的身份验证方式，就像现实生活中的"通行证"。由服务器提供给你一串字符，例如
`eyhHSudhjn351jnSF8j0135j1njoSFHs0813`，这串字符在这个服务器里（或者这个认证域里）就代表你，任何人凭借这串字符串，都可以合理合法的访问你的服务。

:::caution 警告
`Bearer Token`是你在这个服务器中的唯一凭证，因此，在`截图`、`复制`时要小心！

**自己是账号安全的第一责任人**
:::

### 为什么需要 Bearer 认证？

那我们为什么不用账号+密码？非得用Bearer认证呢？因为：

1. **安全性**：避免每次请求都传输用户名密码（而且服务器也不建议保存密码）
2. **无状态**：服务器不需要保存会话
3. **跨域支持**：适合前后端分离架构
4. **权限控制**：不同令牌可拥有不同权限

认证成功后，只需要在`Header`内配置Token，写成诸如：

**`Header: Bearer <秘钥>`即可。**

:::tip 思考
为什么服务器不推荐保存密码？你能否说出原因？
:::

### Bearer认证的构成

#### 1. 令牌（Token）
- **格式**：通常是字符串，可以是 JWT、随机字符串等
- **特性**：自包含或可验证的凭证
- **生命周期**：有明确的过期时间

### Bearer Token 的类型

#### 1. JWT（JSON Web Token）

**结构组成**：

**Header 示例**：
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**Payload 示例：**

```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022,
  "exp": 1516242622,
  "roles": ["user", "admin"]
}
```

:::tip 练习3
请教师启动本手册附带的`jwtserver`，指导同学们完成下列练习：

1. **请你使用`POST`请求，访问`/login`（账号：`test@10z.com`，密码`123456`，你获得了什么？**
> 数据格式为：`{"email": "", "password": ""}`

  - `access_token`是什么？过期时间有多久？
  - 用户名是什么？
  - 你能不能为它再添加一个账号？

2. **请你使用`GET`请求，访问`/test`（需要携带Bearer Token）**
> 这是一个受保护的API，需要你携带上一题中的`Token`

3. **请你使用`POST`请求，访问`/profile`（需要携带Bearer Token），你获得了什么？**
  
  - 如果使用`GET`请求访问，会发生什么？

4. **请你使用`POST`请求，访问`/logout`，登出你的账号**
> 此时，这个密钥会被吊销，而在现实中就相当于你操作了“登出”

  - 再使用第一题获得的`access_token`，访问`/profile`，是什么结果？

5.  _**拓展**_ 【选做】**请你为它编写一个API，用以模拟管理员模仿登录（获得其他人的`Token`）**
> 提示：你可以判断用户的`email`，并修改`login`，若正确且用户的`email`是指定的管理员，则尝试登录其他人的账号

:::

:::caution 注意
本验证服务器未进行安全测试，不要用在正式环境中！
:::

## 示例代码

首先安装必要的库：

```bash
pip install requests
```

GET请求的示例代码：
```python
import requests

def simple_get():
    print("=== 基础 GET 请求 ===")
    response = requests.get('https://httpbin.org/get')
    
    # 检查请求是否成功
    if response.status_code == 200:
        print("请求成功！")
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text[:200]}...")  # 只显示前200个字符
    else:
        print(f"请求失败，状态码: {response.status_code}")

def get_with_params():
    print("\n=== 带参数的 GET 请求 ===")
    
    # 定义查询参数
    params = {
        'name': '张三',
        'age': 25,
        'city': '北京'
    }
    
    response = requests.get('https://httpbin.org/get', params=params)
    
    if response.status_code == 200:
        data = response.json()
        print("请求成功！")
        print(f"请求的URL: {data['url']}")
        print(f"查询参数: {data['args']}")
    else:
        print(f"请求失败")

# 运行 GET 示例
simple_get()
get_with_params()
```

POST请求的示例代码：
```python
import requests
import json

# 表单数据 POST 请求
def post_form_data():
    print("\n=== 表单数据 POST 请求 ===")
    
    # 模拟用户注册数据
    user_data = {
        'username': 'student001',
        'password': 'mypassword123',
        'email': 'student@example.com'
    }
    
    response = requests.post('https://httpbin.org/post', data=user_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ 注册数据提交成功！")
        print(f"服务器收到的数据: {data['form']}")
    else:
        print(f"❌ 提交失败")

# JSON 数据 POST 请求
def post_json_data():
    print("\n=== JSON 数据 POST 请求 ===")
    
    # 创建博客文章数据
    article = {
        'title': '我的第一篇博客',
        'content': '这是博客的内容...',
        'author': '初学者',
        'tags': ['教程', '编程', '学习']
    }
    
    # 使用 json 参数自动设置 Content-Type
    response = requests.post('https://httpbin.org/post', json=article)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ 博客发布成功！")
        print(f"服务器收到的JSON数据: {json.dumps(data['json'], indent=2, ensure_ascii=False)}")
    else:
        print(f"❌ 发布失败")

# 运行 POST 示例
post_form_data()
post_json_data()
```

Bearer认证代码：
```python
import requests

# 基础 Bearer 认证示例
def bearer_auth_basic():
    print("\n=== Bearer 认证基础示例 ===")
    
    # 假设这是从服务器获取的访问令牌
    access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.example.token"
    
    # 设置认证头部
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # 发送带认证的请求
    response = requests.get(
        'https://httpbin.org/bearer',
        headers=headers
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Bearer 认证成功！")
        print(f"认证用户: {data.get('authenticated')}")
        print(f"令牌: {data.get('token')}")
    else:
        print(f"❌ 认证失败，状态码: {response.status_code}")

bearer_auth_basic()
```