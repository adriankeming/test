from django.contrib import admin
from mainsite.models import Post


class PostAdmin(admin.ModelAdmin):
    list_display = ('slug','title','body','pub_date')

admin.site.register(Post, PostAdmin)

