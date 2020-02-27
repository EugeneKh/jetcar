```bash
git clone https://github.com/EugeneKh/jetcar.git
cd jetcar
code .
```
всякие изменения в VSCode + закиньте в эту папку код с разметкой

Чтобы залить изменения - 2 вар.



##№ Вар. I


```bash
git add . //точка нужна
git commit -m «comment»
```
без `-m «myown_comment»` во второй комманде откроется текстовый редактор
- nano, в котором первой строчкой комментарий, потом "Ctrl + O" -> "Enter" -> "Ctrl + X" или
- vim - тогда кнопку POWER на компутере

```bash
git push
```
будет что-то спрашивать, ключи на GitHub. 

## Вар. II
может быть проще прямо в VSCode:

`Ctrl + Shift + G`

![Альтернативный текст](../Аннотация.png)
откроется окошко с авторизацией
