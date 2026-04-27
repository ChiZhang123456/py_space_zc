
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
from matplotlib import gridspec, dates
from pyrfu.pyrf import shock_normal

class shock_gui:
    def __init__(self, B, n, V):
        self.B = B
        self.n = n
        self.V = V
        self.up_range = None
        self.down_range = None
        self.res = None
        self.selection_count = 0

        self.fig = self.init_fig()

        self.span_selector = SpanSelector(
            self.axs[0], self.on_select,
            direction="horizontal", useblit=True,
            props=dict(alpha=0.3, facecolor='blue'),
            interactive=True
        )

        axbtn = self.fig.add_axes([0.87, 0.05, 0.1, 0.05])
        self.button = Button(axbtn, "Compute")
        self.button.on_clicked(self.compute_shock)

        axreset = self.fig.add_axes([0.75, 0.05, 0.1, 0.05])
        self.reset_button = Button(axreset, "Reset")
        self.reset_button.on_clicked(self.reset_selection)

    def init_fig(self):
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(3, 1, hspace=0.2)
        self.axs = [fig.add_subplot(gs[i]) for i in range(3)]

        B = self.B
        self.axs[0].plot(B.time, B[:, 0], label="Bx", color='red')
        self.axs[0].plot(B.time, B[:, 1], label="By", color='blue')
        self.axs[0].plot(B.time, B[:, 2], label="Bz", color='black')
        self.axs[0].legend(loc='upper right', fontsize=9)
        self.axs[0].set_ylabel("B [nT]")

        self.axs[1].plot(self.n.time, self.n.data, color='green')
        self.axs[1].set_ylabel("n [cm⁻³]")

        V = self.V
        self.axs[2].plot(V.time, V[:, 0], label="Vx", color='red')
        self.axs[2].plot(V.time, V[:, 1], label="Vy", color='blue')
        self.axs[2].plot(V.time, V[:, 2], label="Vz", color='black')
        self.axs[2].legend(loc='upper right', fontsize=9)
        self.axs[2].set_ylabel("V [km/s]")
        self.axs[2].set_xlabel("Time")

        all_times = np.concatenate([self.B.time.data, self.n.time.data, self.V.time.data])
        tmin, tmax = np.min(all_times), np.max(all_times)
        for ax in self.axs:
            ax.set_xlim(tmin, tmax)

        fig.suptitle("Select upstream (blue) and downstream (red) intervals", fontsize=12)
        return fig

    def on_select(self, tmin, tmax):
        t1 = np.datetime64(dates.num2date(tmin))
        t2 = np.datetime64(dates.num2date(tmax))

        if self.selection_count == 0:
            self.up_range = (t1, t2)
            print(f"[UP] {t1} to {t2}")
            self.span_selector.set_active(False)
            self.span_selector.disconnect_events()
            del self.span_selector

            self.span_selector = SpanSelector(
                self.axs[0], self.on_select,
                direction="horizontal", useblit=True,
                props=dict(alpha=0.3, facecolor='red'),
                interactive=True
            )
            self.selection_count += 1

        elif self.selection_count == 1:
            self.down_range = (t1, t2)
            print(f"[DOWN] {t1} to {t2}")
            self.selection_count += 1
            self.span_selector.set_active(False)

    def reset_selection(self, event):
        self.up_range = None
        self.down_range = None
        self.selection_count = 0
        self.span_selector.set_active(True)
        self.span_selector.props['facecolor'] = 'blue'
        print("Selections reset.")

    def compute_avg(self, obj, t1, t2):
        t = obj.time.data
        data = obj.data
        mask = (t >= t1) & (t <= t2)
        return np.nanmean(data[mask], axis=0)

    def compute_shock(self, event):
        if self.up_range is None or self.down_range is None:
            print("Please select both upstream and downstream intervals first.")
            return

        try:
            Bu = self.compute_avg(self.B, *self.up_range)
            Bd = self.compute_avg(self.B, *self.down_range)
            Nu = self.compute_avg(self.n, *self.up_range)
            Nd = self.compute_avg(self.n, *self.down_range)
            Vu = self.compute_avg(self.V, *self.up_range)
            Vd = self.compute_avg(self.V, *self.down_range)

            spec = {"b_u": Bu, "b_d": Bd, "n_u": Nu, "n_d": Nd, "v_u": Vu, "v_d": Vd}
            result = shock_normal(spec)
            result["tint_up"] = self.up_range
            result["tint_down"] = self.down_range
            result["b_u"] = Bu
            result["b_d"] = Bd
            result["v_u"] = Vu
            result["v_d"] = Vd
            result["n_u"] = Nu
            result["n_d"] = Nd
            self.res = result
            print("Finish computation")
        except Exception as e:
            print("Error during computation:", e)
