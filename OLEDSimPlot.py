import numpy as np
from Plot import Plot
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.constants import e


class OLEDSimPlot(Plot):

    eps_0 = 8.85 * 10 ** -12  # As/Vm
    eps_r = 3.5
    e=e
    
    @classmethod
    def points(cls, x, dim):
        return int(x // dim)

    def V_0(self, x, wf, b):
        if b * self.offsetCathode == -3 or abs(b) == 2:
            return [self.carrierType * wf if a >= 0 else 0.0 for a in x]
        if b * self.offsetCathode == 3:
            return [
                self.carrierType * wf + self.V * self.e if a >= 0 else 0.0 for a in x
            ]
        else:
            return [self.carrierType * wf if a >= 0 else 0.0 for a in x]

    def V_1(self, x, wf, points, sigma=0.1 * e, mu=0):
        if not self.noGauss:
            z = np.random.normal(mu, sigma, points)
        else:
            z = np.ones(points) * mu
        return [self.carrierType * wf + c if a >= 0 else 0.0 for a, c in zip(x, z)]

    def V_2(self, x, b):
        if b == 0:
            return 0
        if b == -1:
            return 0  # [self.carrierType*(self.e**2)/(16*np.pi*self.eps_0*self.eps_r*(a-x[0])) if a>=0 else 0.0 for a in x]
        if b == 1:
            return 0  # [self.carrierType*(self.e**2)/(16*np.pi*self.eps_0*self.eps_r*(a-x[len(x)-1])) if a>=0 else 0.0 for a in x]

    def V_3(self, x, offset, E=None):
        if E is None:
            E = self.E
        return [
            self.carrierType * self.e * E * (a - offset) if a >= 0 else 0.0 for a in x
        ]

    def __init__(
        self,
        name,
        materials,
        showColAxType=["lin", "lin", "lin"],
        showColAxLim=[None, None, None],
        showColLabel=["", "Depth", "Potential"],
        showColLabelUnit=["", "Depth (nm)", "Potential (eV)"],
        carrierType=1,
        offsetCathode=1,  # 1=true -1=false
        V=0,
        E=None,
        wf_m=-4.2 * e,  # J,
        wf_o=-2.8 * e,  # J,
        Delta=None,
        initpos=0,
        x_scale=10 ** 9,
        x_min=-7,
        metalContactAnode=False,
        metalContactCathode=False,
        colorsMetal=["red", "blue", "green", "cyan", "yellow", "magenta"],
        noGauss=False,
        colorGrad=False,
        alphaBG=0.1,
        alphaMBG=0.3,
        **kwargs,
    ):
        Plot.__init__(
            self,
            name,
            [],
            dataImported=True,
            overrideFileList=True,
            showColAxType=showColAxType,
            showColAxLim=showColAxLim,
            showColLabel=showColLabel,
            showColLabelUnit=showColLabelUnit,
            **kwargs,
        )
        self.materials = materials
        self.carrierType = carrierType
        self.offsetCathode = offsetCathode
        self.V = V
        self.wf_m = wf_m
        self.wf_o = wf_o
        self.initpos = initpos
        self.x_scale = x_scale
        self.x_min = x_min
        self.metalContactAnode = metalContactAnode
        self.metalContactCathode = False
        self.colorsMetal = colorsMetal
        self.noGauss = noGauss
        self.colorGrad = colorGrad
        self.alphaBG = alphaBG
        self.alphaMBG = alphaMBG
        if Delta is None:
            self.Delta = wf_m - wf_o
        else:
            self.Delta = Delta
        if E is None:
            self.E = V / (200 * 10 ** -9)  # V/m; 200nm organics
        else:
            self.E = E
        self.stack = Stack(self.materials, self.colors, self.colorsMetal)
        self.initTex()

    def doPlot(self):
        return self.stack.plotStack(self)


class Material:
    iD = 0

    def __init__(
        self,
        thickness,
        CBLike,
        name="Unknown Material",
        x=0,
        y=0,
        vacY=0,
        dim=10 ** -10,
        metallic=False,
        height=0,
        outsourceDesc=None,
        desc_x_offset=0,
        startID=None,
    ):
        self.name = name
        self.thickness = thickness
        self.CBLike = CBLike
        self.x = x
        self.dim = dim
        self.y = y
        self.vacY = vacY
        self.metallic = metallic
        self.height = height
        self.outsourceDesc = outsourceDesc
        self.desc_x_offset = desc_x_offset
        if startID is None:
            self.id = Material.iD + 1
            Material.iD = Material.iD + 1
        else:
            self.id = startID
            Material.iD = startID

    def __repr__(self):
        return "{}: ({:.0f} nm)".format(self.name, self.thickness * 10 ** 9)

    def __str__(self):
        return self.name

    def name():
        return self.name


class Organic(Material):
    def __init__(
        self,
        thickness,
        CBLike,
        VLLike,
        name="Unknown Material",
        x=0,
        y=0,
        VLDevia=0.05,
        CBDevia=0.05,
        eps_r=3.5,
        metallic=False,
        y_2=0,
        sigma=0.001,
        polarity=0,
        **kwargs,
    ):
        Material.__init__(
            self, thickness, CBLike, name=name, x=x, y=y, metallic=False, **kwargs
        )
        self.VLLike = VLLike
        self.VLDevia = VLDevia
        self.CBDevia = CBDevia
        self.eps_r = eps_r
        self.y_2 = y_2
        self.sigma = sigma
        self.polarity = polarity


class DopedOrganic(Organic):
    def __init__(
        self,
        thickness,
        CBLike,
        VLLike,
        CB2Like,
        VL2Like,
        name2="UnknownMaterial",
        y_3=0,
        y_4=0,
        **kwargs,
    ):
        Organic.__init__(self, thickness, CBLike, VLLike, **kwargs)
        self.CB2Like = CB2Like  # Host
        self.VL2Like = VL2Like  # Host
        self.y_3 = y_3
        self.y_4 = y_4
        self.name2 = name2
        self.id2 = Material.iD + 1
        Material.iD = Material.iD + 1


class Metal(Material):
    def __init__(self, thickness, CBLike, name="Unknown Material", x=0, y=0, **kwargs):
        Material.__init__(
            self, thickness, CBLike, name=name, x=x, y=y, metallic=True, **kwargs
        )


class Stack:
    def __init__(self, Materials, colors=None, colorsMetal=None):
        self.A = [m.metallic for m in Materials]
        self.hil = self.A.index(False)
        self.eil = len(self.A) - self.A[::-1].index(False) - 1
        self.B = (
            [-3]
            + [-2] * (self.hil - 1)
            + [-1]
            + (self.eil - self.hil - 1) * [0]
            + [1]
            + (len(self.A) - 2 - self.eil) * [2]
            + [3]
        )
        self.Materials = Materials
        self.Cz = colors  # ['#1f77b4','#2ca02c','#17becf','#f8e520','#d62728','#ff7f0e','#bcbd22','#9467bd','#8c564b','#e377c2','#7f7f7f']
        self.C = colorsMetal  #

    def __repr__(self):
        return f"Stack: {repr(self.Materials)}"

    def plotStack(self, plot):
        plot.fig, ax = plot.newFig()
        import matplotlib.pyplot as plt

        ax.set_xlabel(plot.showColLabelUnit[plot.xCol])
        ax.set_ylabel(plot.showColLabelUnit[plot.showCol])
        for label in ax.get_yticklabels():
            label.set_fontproperties(plot.default_font)
        curPos = plot.initpos
        for m in self.Materials:
            nexPos = curPos + m.thickness
            m.x = np.linspace(curPos, nexPos, plot.points(m.thickness, m.dim))
            curPos = nexPos
        for m, b in zip(self.Materials, self.B):
            if b * plot.offsetCathode == 3:
                a = m.CBLike
            if b * -plot.offsetCathode == 3:
                c = m.CBLike
            if not m.metallic:
                m.y = (
                    np.asarray(
                        plot.V_1(
                            m.x,
                            m.CBLike,
                            plot.points(m.thickness, m.dim),
                            sigma=m.sigma,
                        )
                    )
                    + np.asarray(plot.V_2(m.x, b))
                    + np.asarray(plot.V_3(m.x, m.x[0]))
                    + np.asarray(plot.V_3(m.x, m.x[0], E=m.polarity))
                )
                m.y_2 = (
                    np.asarray(
                        plot.V_1(
                            m.x,
                            m.VLLike,
                            plot.points(m.thickness, m.dim),
                            sigma=m.sigma,
                        )
                    )
                    + np.asarray(plot.V_2(m.x, b))
                    + np.asarray(plot.V_3(m.x, m.x[0]))
                    + np.asarray(plot.V_3(m.x, m.x[0], E=m.polarity))
                )
                m.vacY = (
                    np.asarray(plot.V_2(m.x, b))
                    + np.asarray(plot.V_3(m.x, m.x[0]))
                    + np.asarray(plot.V_3(m.x, m.x[0], E=m.polarity))
                )
                if not plot.colorGrad:
                    ax.plot(
                        np.asarray(m.x) * plot.x_scale,
                        m.y * e ** -1,
                        color=self.Cz[m.id],
                        label=m.name,
                    )
                    ax.plot(
                        np.asarray(m.x) * plot.x_scale,
                        m.y_2 * e ** -1,
                        color=self.Cz[m.id],
                    )
                    ax.plot(np.asarray(m.x) * plot.x_scale, m.vacY * e ** -1, "k-")
                    if type(m) is DopedOrganic:
                        m.y_3 = (
                            np.asarray(
                                plot.V_1(
                                    m.x,
                                    m.CB2Like,
                                    plot.points(m.thickness, m.dim),
                                    sigma=m.sigma,
                                )
                            )
                            + np.asarray(plot.V_2(m.x, b))
                            + np.asarray(plot.V_3(m.x, m.x[0]))
                            + np.asarray(plot.V_3(m.x, m.x[0], E=m.polarity))
                        )
                        m.y_4 = (
                            np.asarray(
                                plot.V_1(
                                    m.x,
                                    m.VL2Like,
                                    plot.points(m.thickness, m.dim),
                                    sigma=m.sigma,
                                )
                            )
                            + np.asarray(plot.V_2(m.x, b))
                            + np.asarray(plot.V_3(m.x, m.x[0]))
                            + np.asarray(plot.V_3(m.x, m.x[0], E=m.polarity))
                        )
                        ax.plot(
                            np.asarray(m.x) * plot.x_scale,
                            m.y_3 * e ** -1,
                            color=self.Cz[m.id2],
                            label=m.name2,
                        )
                        ax.plot(
                            np.asarray(m.x) * plot.x_scale,
                            m.y_4 * e ** -1,
                            color=self.Cz[m.id2],
                        )
                        ax.fill_between(
                            np.asarray(m.x) * plot.x_scale,
                            m.y_3 * e ** -1,
                            m.y_4 * e ** -1,
                            facecolor=self.Cz[m.id2],
                            interpolate=False,
                            alpha=plot.alphaBG,
                        )
                        ax.fill_between(
                            np.asarray(m.x) * plot.x_scale,
                            m.y_2 * e ** -1,
                            m.y_4 * e ** -1,
                            facecolor=self.Cz[m.id],
                            interpolate=False,
                            alpha=plot.alphaBG,
                        )
                        ax.fill_between(
                            np.asarray(m.x) * plot.x_scale,
                            m.y * e ** -1,
                            m.y_3 * e ** -1,
                            facecolor=self.Cz[m.id],
                            interpolate=False,
                            alpha=plot.alphaBG,
                        )
                    else:
                        ax.fill_between(
                            np.asarray(m.x) * plot.x_scale,
                            m.y * e ** -1,
                            m.y_2 * e ** -1,
                            facecolor=self.Cz[m.id],
                            interpolate=False,
                            alpha=plot.alphaBG,
                        )
                else:
                    self.createGradient(ax, m, plot, b=b)
                    ax.plot(np.asarray(m.x) * plot.x_scale, m.vacY * e ** -1, "k-")
                self.handleOutsourceDesc(m, ax, plot)

        for m, b in zip(self.Materials, self.B):
            if m.metallic:
                if b == -2 and plot.metalContactAnode:
                    m.y = plot.V_0(m.x, c, 3 * -plot.offsetCathode)
                    m.vacY = [d - m.CBLike for d in m.y]
                    ax.plot(
                        np.asarray(m.x) * plot.x_scale,
                        np.asarray(m.y) * e ** -1,
                        color=self.C[b - 1],
                        label=m.name,
                    )
                    ax.fill_between(
                        np.asarray(m.x) * plot.x_scale,
                        x_min,
                        np.asarray(m.y) * e ** -1,
                        interpolate=True,
                        facecolor=self.C[b - 1],
                        alpha=plot.alphaMBG,
                    )
                    ax.text(
                        (m.x[(len(m.x) - 1) // 2] + m.desc_x_offset) * plot.x_scale,
                        (m.y[0] + plot.x_min * e) / 2 * e ** -1 + m.height,
                        m.name,
                        ha="center",
                        va="center",
                    )
                    ax.plot(
                        np.asarray(m.x) * plot.x_scale,
                        np.asarray(m.vacY) * e ** -1,
                        "k-",
                        label="Vacuum",
                    )
                elif b == 2 and plot.metalContactCathode:
                    m.y = plot.V_0(m.x, a, 3 * plot.offsetCathode)
                    m.vacY = [d - m.CBLike for d in m.y]
                    ax.plot(
                        np.asarray(m.x) * plot.x_scale,
                        np.asarray(m.y) * e ** -1,
                        color=self.C[b - 1],
                        label=m.name,
                    )
                    ax.fill_between(
                        np.asarray(m.x) * plot.x_scale,
                        plot.x_min,
                        np.asarray(m.y) * e ** -1,
                        interpolate=True,
                        facecolor=self.C[b - 1],
                        alpha=plot.alphaMBG,
                    )
                    ax.text(
                        (m.x[(len(m.x) - 1) // 2] + m.desc_x_offset) * plot.x_scale,
                        (m.y[0] + plot.x_min * e) / 2 * e ** -1 + m.height,
                        m.name,
                        ha="center",
                        va="center",
                    )
                    ax.plot(
                        np.asarray(m.x) * plot.x_scale,
                        np.asarray(m.vacY) * e ** -1,
                        "k-",
                        label="Vacuum",
                    )
                else:
                    m.y = plot.V_0(m.x, m.CBLike, b)
                    m.vacY = plot.V_0(m.x, 0, b)
                    if b == -2:
                        ax.plot(
                            np.asarray(m.x) * plot.x_scale + 0.25,
                            np.asarray(m.y) * e ** -1,
                            color=self.C[b - 1],
                            label=m.name,
                        )
                        ax.fill_between(
                            np.hstack([m.x, [m.x[-1] + 0.5 / plot.x_scale]])
                            * plot.x_scale,
                            plot.x_min,
                            np.hstack([m.y, [m.y[-1]]]) * plot.e ** -1,
                            interpolate=True,
                            facecolor=self.C[b - 1],
                            alpha=plot.alphaMBG,
                        )
                    elif b == 2:
                        ax.plot(
                            np.asarray(m.x) * plot.x_scale - 0.25,
                            np.asarray(m.y) * e ** -1,
                            color=self.C[b - 1],
                            label=m.name,
                        )
                        ax.fill_between(
                            np.hstack([[m.x[0] - 0.5 / plot.x_scale], m.x])
                            * plot.x_scale,
                            plot.x_min,
                            np.hstack([[m.y[0]], m.y]) * plot.e ** -1,
                            interpolate=True,
                            facecolor=self.C[b - 1],
                            alpha=plot.alphaMBG,
                        )
                    else:
                        ax.plot(
                            np.asarray(m.x) * plot.x_scale,
                            np.asarray(m.y) * e ** -1,
                            color=self.C[b - 1],
                            label=m.name,
                        )
                        ax.fill_between(
                            np.asarray(m.x) * plot.x_scale,
                            plot.x_min,
                            np.asarray(m.y) * plot.e ** -1,
                            interpolate=True,
                            facecolor=self.C[b - 1],
                            alpha=0.3,
                        )
                    ax.text(
                        (m.x[(len(m.x) - 1) // 2] + m.desc_x_offset) * plot.x_scale,
                        (m.y[0] + plot.x_min * e) / 2 * e ** -1 + m.height,
                        m.name,
                        ha="center",
                        va="center",
                    )
                    ax.plot(
                        np.asarray(m.x) * plot.x_scale,
                        np.asarray(m.vacY) * plot.e ** -1,
                        "k-",
                    )
        if plot.titleBool:
            ax.set_title(plot.title, fontsize=plot.titleFontSize)
        ax.set_ylim(plot.x_min, plot.V + 1)
        if plot.axXLim is not None:
            ax.set_xlim(*plot.axXLim)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.grid(which="major", ls=":", alpha=0.5, axis="y")
        plot.saveFig()
        return [plot, plot.processFileName(option=".pdf")]  # filename

    def handleOutsourceDesc(self, m, ax, plot):
        if m.outsourceDesc is None:
            if type(m) is DopedOrganic:
                ax.text(
                    m.x[(len(m.x) - 1) // 2] * plot.x_scale,
                    (m.y_4[(len(m.x) - 1) // 2] + m.y_3[(len(m.x) - 1) // 2])
                    / 2
                    * e ** -1
                    + m.height,
                    m.name + ":" + m.name2,
                    ha="center",
                    va="center",
                )
            else:
                ax.text(
                    m.x[(len(m.x) - 1) // 2] * plot.x_scale,
                    (m.y_2[(len(m.x) - 1) // 2] + m.y[(len(m.x) - 1) // 2])
                    / 2
                    * e ** -1
                    + m.height,
                    m.name,
                    ha="center",
                    va="center",
                )
        else:
            if type(m) is DopedOrganic:
                ax.annotate(
                    m.name + ":" + m.name2,
                    xy=(
                        m.x[(len(m.x) - 1) // 2] * plot.x_scale,
                        (m.y_4[(len(m.x) - 1) // 2] + m.y_3[(len(m.x) - 1) // 2])
                        / 2
                        * e ** -1
                        + m.height,
                    ),
                    xytext=(m.x[(len(m.x) - 1) // 2] * plot.x_scale, m.outsourceDesc),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    ha="center",
                    va="center",
                )
            else:
                ax.annotate(
                    m.name,
                    xy=(
                        m.x[(len(m.x) - 1) // 2] * plot.x_scale,
                        (m.y_2[(len(m.x) - 1) // 2] + m.y[(len(m.x) - 1) // 2])
                        / 2
                        * e ** -1
                        + m.height,
                    ),
                    xytext=(m.x[(len(m.x) - 1) // 2] * plot.x_scale, m.outsourceDesc),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    ha="center",
                    va="center",
                )

    def createGradient(self, ax, m, plot, b=0):
        def gauss(x, mu, amp, sigma):
            return amp * np.exp(-((x - mu) ** 2 / sigma))

        def gradient_image(ax, extent, direction=0, cmap_range=(0, 1), **kwargs):
            """
            Draw a gradient image based on a colormap.

            Parameters
            ----------
            ax : Axes
                The axes to draw on.
            extent
                The extent of the image as (xmin, xmax, ymin, ymax).
                By default, this is in Axes coordinates but may be
                changed using the *transform* kwarg.
            direction : float
                The direction of the gradient. This is a number in
                range 0 (=vertical) to 1 (=horizontal).
            cmap_range : float, float
                The fraction (cmin, cmax) of the colormap that should be
                used for the gradient, where the complete colormap is (0, 1).
            **kwargs
                Other parameters are passed on to `.Axes.imshow()`.
                In particular useful is *cmap*.
            """
            phi = direction * np.pi / 2
            v = np.array([np.cos(phi), np.sin(phi)])
            X = np.array([[v @ [1, 0], v @ [1, 1]], [v @ [0, 0], v @ [0, 1]]])
            a, b = cmap_range
            X = a + (b - a) / X.max() * X
            im = ax.imshow(
                X, extent=extent, interpolation="bicubic", vmin=0, vmax=1, **kwargs
            )
            return im

        h = self.Cz[m.id].lstrip("#")
        rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
        new_colors = gauss(np.linspace(0, 1, 256), 0.5, 1, 0.05)
        # new_colors[127:]=np.amax(np.asarray([new_colors[127:],plot.alphaBG*np.ones(129)]), axis=0)
        new_pre_cmp = np.array(
            [
                [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, new_color]
                for new_color in new_colors
            ]
        )
        new_cmp = ListedColormap(new_pre_cmp)
        extent1 = (
            m.x[0] * plot.x_scale,
            m.x[-1] * plot.x_scale,
            (m.y[1] - m.sigma) * e ** -1,
            (m.y[1] + m.sigma) * e ** -1,
        )
        extent2 = (
            m.x[0] * plot.x_scale,
            m.x[-1] * plot.x_scale,
            (m.y_2[1] - m.sigma) * e ** -1,
            (m.y_2[1] + m.sigma) * e ** -1,
        )
        gradient_image(ax, extent=extent1, cmap=new_cmp, cmap_range=(0, 1))
        gradient_image(ax, extent=extent2, cmap=new_cmp, cmap_range=(1, 0))
        if type(m) is DopedOrganic:
            h_2 = self.Cz[m.id2].lstrip("#")
            rgb_2 = tuple(int(h_2[i : i + 2], 16) for i in (0, 2, 4))
            new_colors_2 = gauss(np.linspace(0, 1, 256), 0.5, 1, 0.05)
            # new_colors[127:]=np.amax(np.asarray([new_colors[127:],plot.alphaBG*np.ones(129)]), axis=0)
            new_pre_cmp_2 = np.array(
                [
                    [rgb_2[0] / 255, rgb_2[1] / 255, rgb_2[2] / 255, new_color]
                    for new_color in new_colors_2
                ]
            )
            new_cmp_2 = ListedColormap(new_pre_cmp_2)
            m.y_3 = (
                np.asarray(
                    plot.V_1(
                        m.x, m.CB2Like, plot.points(m.thickness, m.dim), sigma=m.sigma
                    )
                )
                + np.asarray(plot.V_2(m.x, b))
                + np.asarray(plot.V_3(m.x, m.x[0]))
                + np.asarray(plot.V_3(m.x, m.x[0], E=m.polarity))
            )
            m.y_4 = (
                np.asarray(
                    plot.V_1(
                        m.x, m.VL2Like, plot.points(m.thickness, m.dim), sigma=m.sigma
                    )
                )
                + np.asarray(plot.V_2(m.x, b))
                + np.asarray(plot.V_3(m.x, m.x[0]))
                + np.asarray(plot.V_3(m.x, m.x[0], E=m.polarity))
            )
            extent3 = (
                m.x[0] * plot.x_scale,
                m.x[-1] * plot.x_scale,
                (m.y_3[1] - m.sigma) * e ** -1,
                (m.y_3[1] + m.sigma) * e ** -1,
            )
            extent4 = (
                m.x[0] * plot.x_scale,
                m.x[-1] * plot.x_scale,
                (m.y_4[1] - m.sigma) * e ** -1,
                (m.y_4[1] + m.sigma) * e ** -1,
            )
            gradient_image(ax, extent=extent3, cmap=new_cmp_2, cmap_range=(0, 1))
            gradient_image(ax, extent=extent4, cmap=new_cmp_2, cmap_range=(1, 0))
            ax.fill_between(
                np.asarray(m.x) * plot.x_scale,
                m.y_3 * e ** -1,
                m.y_4 * e ** -1,
                facecolor=self.Cz[m.id2],
                interpolate=False,
                alpha=plot.alphaBG,
            )
            ax.fill_between(
                np.asarray(m.x) * plot.x_scale,
                m.y_2 * e ** -1,
                m.y_4 * e ** -1,
                facecolor=self.Cz[m.id],
                interpolate=False,
                alpha=plot.alphaBG,
            )
            ax.fill_between(
                np.asarray(m.x) * plot.x_scale,
                m.y * e ** -1,
                m.y_3 * e ** -1,
                facecolor=self.Cz[m.id],
                interpolate=False,
                alpha=plot.alphaBG,
            )
        else:
            ax.fill_between(
                np.asarray(m.x) * plot.x_scale,
                (m.y) * e ** -1,
                (m.y_2) * e ** -1,
                facecolor=self.Cz[m.id],
                interpolate=False,
                alpha=plot.alphaBG,
            )
        ax.set_aspect("auto")
