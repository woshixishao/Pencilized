from PIL import Image
from black import draw_black
from color import draw_color

imr = Image.open("t.jpg")
img_b = draw_black(imr, gammaS=1, gammaI=1)
img_b = img_b.convert('RGB')
img_b.save('t_b.jpg')
img_c = draw_color(imr, gammaS=1, gammaI=1)
img_c = img_c.convert('RGB')
img_c.save('t_c.jpg')
img_c.show()
