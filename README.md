# image_preprocessing

При обработке документов часто приходится иметь дело с изобаржениями плохого качества.  
Для улучшения распознования текста, необходимо сделать текст более читаемым.  
Одно из решений:  
Бинаризовать изражение(темный текст на светлом фоне), используя adaptiveThreshold, затем воспользуемся морфологическими преобразованиями Erosion, Dilation или Opening (Dilation + Erosion).

Исходное изображение:
![1](https://user-images.githubusercontent.com/56885818/204333239-3f100c75-3cb9-4960-a393-d12d5d3e56d7.jpg)


Изображение на выходе:
![11](https://user-images.githubusercontent.com/56885818/204333654-0997d5a4-c53b-45a5-bfbc-895d0b12ab78.jpg)
:+1:
