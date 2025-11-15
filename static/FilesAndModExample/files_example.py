
class Myfile:
    FileName = None
    def __init__(self , File_Name):
        self.FileName = File_Name
    def ReadMyFiles(self):
        with open(self.FileName, mode='r') as f:
            message = f.read()
        print(message)
    def WriteMyFiles(self , message):
        #使用a模式避免被覆盖，或者可以手动控制指针
        with open(self.FileName, mode='a') as f:
            f.write(message)