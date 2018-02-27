
from django import forms

class NaverMapPointWidget(forms.TextInput):
    def render(self, name, value, attrs):
        html = '''
            <div style="width: 100px; height: 100px; background-color: red;">
                네이버 지도를 그려봅시다.
            </div>
        '''
        parent_html = super().render(name, value, attrs)
        return parent_html + html