import os
from django.conf import settings
from django.views.generic import View, TemplateView
from django.http import HttpResponse, JsonResponse

class PostListView1(View):
    '''Class Based View(CBV): 직접 문자열로 HTML 형식 응답하기'''
    def get(self, request):
        name = '최종현'
        html = self.get_template_string().format(name=name)
        # html = '''<h1>AskDjango</h1>
        #         <p>{name}</p>
        #         <p>Django 공부 열심히 할게요</p>
        #         '''.format(name=name)
        return HttpResponse(html)

    def get_template_string(self):
        return '''<h1>AskDjango</h1>
                <p>{name}</p>
                <p>Django 공부 열심히 할게요</p>
                '''

post_list1 = PostListView1.as_view()


class PostListView2(TemplateView):
    '''CBV: 템플릿을 통해 HTML 형식 응답하기'''
    template_name = 'dojo/post_list.html'

    def get_context_data(self):
        context = super().get_context_data()
        context['name'] = '최종현'
        return context

post_list2 = PostListView2.as_view()


class PostListView3(View):
    '''CBV: JSON 형식 응답하기'''

    def get(self, request):
        return JsonResponse(self.get_data(), json_dumps_params={'ensure_ascii': False})

    def get_data(self):
        return {
            'message': '안녕, 파이썬 & 장고',
            'items': ['파이썬', '장고', 'Celery', 'Azure', 'AWS']
        }

post_list3 = PostListView3.as_view()


class ExcelDownloadView(View):
    '''CBV: 엑셀 다운로드 응답하기'''

    excel_path = os.path.join(settings.BASE_DIR, 'test.xlsx')

    def get(self, request):
        filename = os.path.basename(self.excel_path) # 파일명 가져오기
        with open(self.excel_path, 'rb') as f:
            response = HttpResponse(f, content_type='application/vnd.ms-excel')
            # 필요한 응답헤더 세팅
            response['Content-Disposition'] = 'attachment; filename="{}"'.format(filename)
            return response

excel_download = ExcelDownloadView.as_view()
