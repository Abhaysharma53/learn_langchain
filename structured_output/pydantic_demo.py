from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal

class Student(BaseModel):
    name: str = 'Abhay'
    age: Optional[int] = 'None'
    cgpa: float = Field(gt = 0, lt = 10, description = "CGPA of the student", default = 7)
    email: EmailStr

new_student = {'email': "abc@error.com"}

student = Student(**new_student)
print(student)
print(type(student))
print(student.email)

#typecasting pydantic object to dictionary
stu_dict = dict(student)
print(type(stu_dict))
print(stu_dict['email'])