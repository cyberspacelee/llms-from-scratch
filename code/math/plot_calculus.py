"""重建微积分章节的数值图；轨迹均由梯度更新计算。"""
import numpy as np
from plot_style import plt, configure, save, GREEN, ORANGE, BLUE, INK, MUTED


def main():
    configure()
    x = np.linspace(-2, 3, 400)
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    ax.plot(x, x*x, color=GREEN, label=r"函数 $f(x)=x^2$")
    ax.plot(x, 2*x-1, "--", color=ORANGE, label=r"在 $x=1$ 处的切线：$2x-1$")
    ax.scatter([1], [1], color=ORANGE, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=.8)
    ax.axvline(0, color=MUTED, linewidth=.8)
    ax.set(xlabel="输入 x", ylabel="函数值 f(x)", ylim=(-2, 9), title="导数：接触点附近的切线斜率")
    ax.legend()
    ax.grid(alpha=.15)
    save(fig, "calculus-tangent.svg")

    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 250), np.linspace(-1.8, 1.8, 250))
    fig, ax = plt.subplots(figsize=(8, 5.4), layout="constrained")
    contours = ax.contour(xx, yy, .5*(xx**2+4*yy**2), levels=[.1,.3,.6,1,2,3,4,6], colors=BLUE, alpha=.5)
    ax.clabel(contours, fontsize=9)
    for eta, color in [(0.2,GREEN),(0.45,ORANGE)]:
        point = np.array([2.,1.])
        path = [point.copy()]
        for _ in range(12):
            point = point - eta * np.array([point[0],4*point[1]])
            path.append(point.copy())
        path = np.array(path)
        ax.plot(path[:,0],path[:,1],"o-",color=color,markersize=4,label=rf"学习率 $\eta={eta}$")
    ax.annotate("起点 (2, 1)",xy=(2,1),xytext=(1.3,1.4),arrowprops={"arrowstyle":"->"})
    ax.scatter([0],[0],marker="*",s=120,color=INK,label="最小值点")
    ax.set(xlabel=r"第一坐标 $x_1$",ylabel=r"第二坐标 $x_2$",title=r"等高线与梯度下降：$L=\frac{1}{2}(x_1^2+4x_2^2)$",aspect="equal")
    ax.legend(loc="lower left")
    save(fig, "calculus-descent.svg")
    print("已生成微积分函数图")


if __name__ == "__main__":
    main()
