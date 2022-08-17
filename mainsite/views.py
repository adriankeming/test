from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Post
from datetime import datetime

# Create your views here.
def homepage(request):
    posts = Post.objects.all()
    post_lists = list()
    now = datetime.now()
    return render(request, 'index.html', locals())
    # for count, post in enumerate(posts):
    #     post_lists.append("No.{}:".format(str(count)) + str(post)+"<br>") 
    #     post_lists.append("<small>"+str(post.body)+"</small><br><br>")
    # return HttpResponse(post_lists)

def showpost(request, slug):
    try:
        post = Post.objects.get(slug = slug)
        if post != None:
            return render(request, 'post.html', locals())
    except:
        return redirect('/')
