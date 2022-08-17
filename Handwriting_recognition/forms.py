# from django import forms


# class ImageUploadForm(forms.Form):
#     image = forms.ImageField()

from django import forms
from Handwriting_recognition.models import upload_img

class ImageFileUploadForm(forms.ModelForm):
    
    class Meta:
        model = upload_img
        fields = ('請輸入員工編號', '選擇檔案')
