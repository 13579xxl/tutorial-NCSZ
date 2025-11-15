class MyClass:
    __my_name = None
    def __init__(self , my_name):
        self.__my_name = my_name
    def PrintMyName(self):
        print(self.__my_name)
#main
print("Please input your name")
name = input() 
My_Class = MyClass(name)
My_Class.PrintMyName()
