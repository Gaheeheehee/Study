
from django import forms
from django.conf import settings  # = django/conf/global_settings.py + askdjango/settings.py
from django.template.loader import render_to_string

class NaverMapPointWidget(forms.TextInput):
    def render(self, name, value, attrs):
        context = {
            'naver_client_id': settings.NAVER_CLIENT_ID,
        }
        html = render_to_string('widgets/naver_map_point_widget.html', context)
        parent_html = super().render(name, value, attrs)
        return parent_html + html