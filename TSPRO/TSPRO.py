from tkinter import *
from tkinter.filedialog import askdirectory, Open, asksaveasfilename
import matplotlib.pyplot as plt
import pandas as pd
import os

import velocities
import constants

# background of selected paths
COLOR_PATHS = "#DCDCDC"

class TSPRO(Tk):

    def __init__(self, *args, **kwargs):
        
        Tk.__init__(self, *args, **kwargs)
        Tk.wm_title(self, "TSPRO")
        container = Frame(self)

        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.container = container

        self.init_menu_window()

        self.frames = {}
        for F in (StartPage, Fod2Disco):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)


    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


    def init_menu_window(self):    
        menubar = Menu(self.container)
        # file menu
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save settings", command = lambda: print('Nothing yet'))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=lambda: print('Nothing yet'))
        menubar.add_cascade(label="File", menu=filemenu)

        # create the Options object
        optionsmenu = Menu(self.container)
        optionsmenu.add_command(label="FOD2DISCO", command=lambda: self.show_frame(Fod2Disco) )
        menubar.add_cascade(label="Options", menu=optionsmenu)

        Tk.config(self, menu=menubar)


class Fod2Disco(Frame):

    def __init__(self, master, controller):
        Frame.__init__(self, master)

        self.init_ask_files()
        self.init_run_buttons(controller)

    def init_ask_files(self):
        label = Label(self, text="Create DISCO file from Bernese FODITS file",  font='bold')
        label.pack(pady=10,padx=10)

        frame0 = Frame(self, borderwidth=1, relief=GROOVE)
        frame0.pack(fill=X)

        # load FODITS file
        frame01 = Frame(frame0)
        frame01.pack(fill=X)
        label2 = Label(frame01, text="Load FODITS output from Bernese 5.2")
        label2.pack(pady=10,padx=10)
        b_Bernfile = Button(frame01, text='Load File', command=self.load_Bern_file, width=12)
        b_Bernfile.pack(fill=X, side=LEFT, padx=5)
        self.label_Bernfile = Label(frame01, text="No file selected", anchor=W, bg=COLOR_PATHS)
        self.label_Bernfile.pack(fill=X, padx=5, expand=True, side=LEFT)

        # save DISCO file
        frame02 = Frame(frame0)
        frame02.pack(fill=X)
        label3 = Label(frame02, text="Save created DISCO file")
        label3.pack(pady=10,padx=10)
        b_save = Button(frame02, text='Save file as', command=self.save_new_DISCO, width=12)
        b_save.pack(fill=X, side=LEFT, padx=5)
        self.label_savedsc = Label(frame02, text="No file selected", anchor=W, bg=COLOR_PATHS)
        self.label_savedsc.pack(fill=X, padx=5, expand=True)

        # add free line only
        frame3 = Frame(frame0, borderwidth=1)
        frame3.pack(fill=X)
        Label(frame3, text="").pack(fill=X, padx=5, expand=True)

    def init_run_buttons(self, controller):
        frame0 = Frame(self, borderwidth=1, relief=GROOVE)
        frame0.pack(fill=X)

        self.convert_but = Button(frame0, text="Run convertion",
                            command=self.convertion)
        self.convert_but.pack(side=LEFT, padx=50)
        self.convert_but.config(state='disabled')

        back_button = Button(frame0, text="Back to Main page",
                            command=lambda: controller.show_frame(StartPage))
        back_button.pack(side=RIGHT, padx=50)

    def load_Bern_file(self):
        dlg = Open(self)
        bfile = dlg.show()
        self.label_Bernfile['text'] = bfile
        self.convert_but.config(state='normal')

    def save_new_DISCO(self):
            self.new_dst = asksaveasfilename(defaultextension=".dst")
            self.label_savedsc['text'] = self.new_dst

    def convertion(self):
         # dates of discontinuities for all stations from bernese
        bern_discons = velocities.get_Bern_discontinuities(self.label_Bernfile['text'])
        # write new  DISCO file with dates of discontinuities
        velocities.create_custom_discofile(self.new_dst, bern_discons)


class StartPage(Frame):

    def __init__(self, master, controller):
        Frame.__init__(self, master)                

        self.init_open_files()
        self.init_estimation_options()
        self.init_listbox()
        self.init_save_options()
        self.init_run_buttons()
        self.init_text_window()

    def init_open_files(self):

        def callback_dis():
            ftypes = [('Disco files', '*.dst'), ('All files', '*')]
            dlg = Open(self, filetypes = ftypes)
            fl = dlg.show()
            self.label_discofile['text'] = fl
            # load stations to the listbox
            discos_dates = velocities.load_discofile(fl)
            all_stations = sorted(list(discos_dates))

            self.lb.configure(state=NORMAL)
            self.lb.delete(0,END)
            for st in all_stations:
                self.lb.insert(END, st)

            # disable listbox again if multistation mode
            if self.solution_mode.get() == 0:
                self.lb.configure(state='disabled')
  
        def callback_excl():
            ftypes = [('Excl files', '*.excl'), ('All files', '*')]
            dlg = Open(self, filetypes = ftypes)
            fl = dlg.show()
            self.label_exclfile['text'] = fl
            # save used paths to file
            self.write_input_hist(self.label_exclfile['text'], self.label_ddir['text'])

        def callback_dir():
            dirname = askdirectory() 
            self.label_ddir['text'] = dirname
            self.write_input_hist(self.label_exclfile['text'], self.label_ddir['text'])
 

        frame1 = Frame(self, borderwidth=1)
        frame1.pack(fill=X)
        Label(frame1, text="INPUT FILES", font='bold').pack(fill=X, padx=5, expand=True)
        b_discofile = Button(frame1, text='Disco File', command=callback_dis, width=12)
        b_discofile.pack(fill=X, side=LEFT, padx=5)
        self.label_discofile = Label(frame1, text="No file selected", anchor=W, bg=COLOR_PATHS)
        self.label_discofile.pack(fill=X, padx=5, expand=True)

        frame2 = Frame(self, borderwidth=1)
        frame2.pack(fill=X)
        b_exclfile = Button(frame2, text='Exclusion File', command=callback_excl, width=12)
        b_exclfile.pack(fill=X, side=LEFT, padx=5)
        self.label_exclfile = Label(frame2, text="No file selected", anchor=W, bg=COLOR_PATHS)
        self.label_exclfile.pack(fill=X, padx=5, expand=True, side=LEFT)

        frame3 = Frame(self, borderwidth=1)
        frame3.pack(fill=X)
        b_ddir = Button(frame3, text='Data Directory', command=callback_dir, width=12)
        b_ddir.pack(fill=X, side=LEFT, padx=5)
        self.label_ddir = Label(frame3, text="No directory selected", anchor=W, bg=COLOR_PATHS)
        self.label_ddir.pack(fill=X, padx=5, expand=True, side=LEFT)

        # predefine last used paths
        if os.path.exists(constants.HIST_FILE):
            with open(constants.HIST_FILE) as obj:
                excl_pre, ddir_pre = obj.readlines()
            self.label_exclfile['text'] = excl_pre.strip()
            self.label_ddir['text'] = ddir_pre

        # add free line only
        frame3 = Frame(self, borderwidth=1)
        frame3.pack(fill=X)
        Label(frame3, text="").pack(fill=X, padx=5, expand=True)

    def init_estimation_options(self):
        # submaster frame
        frame0 = Frame(self, borderwidth=1, relief=GROOVE)
        frame0.pack(fill=X)

        # headline
        frame_headline = Frame(frame0)
        frame_headline.pack(fill=X)
        head = Label(frame_headline, text="ESTIMATION OPTIONS", font='bold')
        head.pack(padx=5, side=LEFT, expand=True, anchor=CENTER)
        head.place(relx=0.5, rely=0.5, anchor=CENTER)
        # info button
        info_b = Button(frame_headline, text ="question", relief=RAISED,\
                         bitmap="question", command=self.outlier_info)
        info_b.pack(expand=False, anchor=E, padx=90)

        # left frame with period and outliers options
        frame_left = Frame(frame0)
        frame_left.pack(fill=X, side=LEFT)
        var_year = IntVar()
        chk_year = Checkbutton(frame_left, text='Period-yearly', variable=var_year)
        chk_year.pack(anchor=W, expand=YES)

        # prior sigmas as weights checkbutton
        var_weights = IntVar()
        chk_weight = Checkbutton(frame_left, text='WLS', variable=var_weights)
        chk_weight.pack(anchor=W, expand=YES)

        # Outlier checkboutton and sigma entry 
        self.e_out = Entry(frame_left, width=3)
        self.e_out.insert(END, '3')
        self.e_out.configure(state='disabled')
        var_out = IntVar()
        chk_out = Checkbutton(frame_left, text='Outliers', variable=var_out, command=lambda: self.activate(self.e_out,var_out))
        chk_out.pack(side=LEFT, anchor=W, expand=YES)
        self.e_out.pack(side=LEFT, expand=YES, padx=30, anchor=E)

        self.chbuts = [var_year, var_weights, var_out]

        # right frame with time exclusion options
        frame_right = Frame(frame0)
        frame_right.pack(fill=X, side=RIGHT)
        Label(frame_right, text="Exclude TS shorter then (years):").pack(padx=20, expand=True)

        frame_right1 = Frame(frame_right)
        frame_right1.pack(fill=X)
        frame_right2 = Frame(frame_right)
        frame_right2.pack(fill=X)

        Label(frame_right1, text="Full TS").pack(side=LEFT, expand=True, anchor=E)
        self.e_exl_full = Entry(frame_right1, width=4)
        self.e_exl_full.pack(side=LEFT, expand=YES, anchor=CENTER)
        self.e_exl_full.insert(END, '3')
        Label(frame_right2, text="Sub TS").pack(side=LEFT, expand=True, anchor=E)
        self.e_exl_sub = Entry(frame_right2, width=4)
        self.e_exl_sub.pack(side=LEFT, expand=YES, anchor=CENTER)
        self.e_exl_sub.insert(END, '1.25')

    def init_listbox(self):
        frame0 = Frame(self, borderwidth=1, relief=GROOVE)
        frame0.pack(fill=X)

        frame1 = Frame(frame0)
        frame1.pack(fill=X)
        Label(frame1, text="STATIONS IN DISCO FILE", font='bold').pack(fill=X, padx=5, expand=True)

        self.solution_mode = IntVar()
        self.solution_mode.set(0)
        r1 = Radiobutton(frame1, text="Compute all stations", padx = 20, variable=self.solution_mode, value=0, command=lambda: self.activate(self.lb,self.solution_mode))
        r1.pack(anchor=W, side=LEFT)

        r2 = Radiobutton(frame1, text="Compute one station", padx = 20, variable=self.solution_mode, value=1, command=lambda: self.activate(self.lb,self.solution_mode))
        r2.pack(anchor=W, side=LEFT)

        frame2 = Frame(frame0)
        frame2.pack(fill=X)
        self.lb = Listbox(frame2)

        self.lb.bind("<<ListboxSelect>>", self.onSelect)
        self.lb.pack(pady=5)
        self.lb.configure(state='disabled')

    def init_save_options(self):

        def callback_res():
            self.savepath = asksaveasfilename(defaultextension=".csv")
            print(self.savepath)
            self.label_resfile['text'] = self.savepath

        self.savepath = ''

        frame0 = Frame(self, borderwidth=1, relief=GROOVE)
        frame0.pack(fill=X)
        Label(frame0, text="SAVE OPTIONS (optional)", font='bold').pack(fill=X, padx=5, expand=True)

        frame01 = Frame(frame0)
        frame01.pack(fill=X)
        b_save = Button(frame01, text='Save residuals as', command=callback_res, width=14)
        b_save.pack(fill=X, side=LEFT, padx=5)
        self.label_resfile = Label(frame01, text="No file selected", anchor=W, bg=COLOR_PATHS)
        self.label_resfile.pack(fill=X, padx=5, expand=True)

        # for free line only
        frame3 = Frame(frame0)
        frame3.pack(fill=X)
        Label(frame3, text="").pack(fill=X, padx=5, expand=True)

    def init_run_buttons(self):
        frame1 = Frame(self, borderwidth=1, relief=GROOVE)
        frame1.pack(fill=X)
        runButton = Button(frame1, text="Run", command=self.run)
        runButton.pack()

    def init_text_window(self):
        frame1 = Frame(self)
        frame1.pack(fill=X)
        S = Scrollbar(frame1)
        self.text_window = Text(frame1, height=4, width=55)
        self.text_window.pack(side=LEFT, fill=Y)
        S.pack(side=LEFT, fill=Y)
        S.config(command=self.text_window.yview)
        self.text_window.config(yscrollcommand=S.set)

    def activate(self, elem, var):
        if var.get() == 0:
            elem.configure(state='disabled')
        else:
            elem.configure(state='normal')

    def onSelect(self, val):
        sender = val.widget
        idx = sender.curselection()
        try:
            value = sender.get(idx)
            self.sel_station = value
        except:
            pass

    def write_input_hist(self, file, datadir):
        ''' Write file with used input paths'''
        with open(constants.HIST_FILE, 'w') as obj:
            obj.write(file + '\n')
            obj.write(datadir)

    def outlier_info(self):
        toplevel = Toplevel()
        text = '''
        ------Period-yearly------
        Po zaskrtnuti checkboxu bude v modeli uvazovana rocna perioda.

        ------WLS------
        Po zaskrtnuti checkboxu bude pri odhade rychlosti uvazovana presnost 
        vstupnych dat (observacii) vo forme matice vah P.

        ------Outliers------
        Po zaskrtnuti checkboxu "Outliers" budu pri vypocte odstranene odlahle hodnoty.
        Odlahle hodnoty budu definovane prekrocenim zvoleneho nasobku RMSE modelu 
        (s n-k stupnami volnosti) od odhadnutych hodnot. 
        Moznost volby nasobku RMSE sa aktivuje po zaskrtnuti checkboxu "Outliers" 
        a defaultna hodnota je 3.

        ------Full TS------
        Vloz hodnotu v rokoch napr. 2.5.
        Ak cekova dlzka casoveho radu bude kratsia ako vlozena hodnota v rokoch, tak bude 
        tento casovy rad vo vypocte ignorovany.
        Ak casovy rad obsahuje skoky definovane v subore *.dst, tak je celkova dlzka definovana 
        ako sucet dlzok akceptovanych ciastkovych casovych radov (oddelenych datummi skokov). 

        ------Sub TS------
        Vloz hodnotu v rokoch napr. 1.5.
        Ak casovy rad obsahuje skoky, tak ciastkove casove rady oddelene datumami skokov, 
        ktore su kratsie ako vlozena hodnota v rokoch, budu vylucene z vypoctu.

                '''
        label1 = Label(toplevel, text=text, height=0, width=80, justify=LEFT,)
        label1.pack(fill=X)

    def run(self):
        COORS_FILE = constants.COORS_FILE
        SHP_FILE = constants.BORDERS_SHP
        DST_FILE = self.label_discofile['text']
        EXCL_FILE = self.label_exclfile['text']
        TS_DIR = self.label_ddir['text']

        # 0 - period, 1 - outliers
        fit_options = [j.get()==1 for j in self.chbuts]
        period_year = fit_options[0]
        WLS = fit_options[1]
        excl_outliers = fit_options[2]
        if excl_outliers:
            outlier_trash = float(self.e_out.get())
        else:
            outlier_trash = None

        # time threshold for exclusion of short TSs
        full_tresh_year = float(self.e_exl_full.get())
        sub_tresh_year = float(self.e_exl_sub.get())
        full_tresh_weeks = int(full_tresh_year*52)
        sub_tresh_weeks = int(sub_tresh_year*52)

        save_path = self.savepath

        plt.ion()
        # multistation mode
        if self.solution_mode.get() == 0:
            stations_velos, SDs, residuals = velocities.get_final_velocities(DST_FILE, EXCL_FILE, TS_DIR, plot_each_fit=False, 
                                            plot_each_outliers=False, period_year=period_year, stat=None, outlier_trash=outlier_trash,
                                            weeks_tresh_part=sub_tresh_weeks, weeks_tresh_all=full_tresh_weeks, use_weights=WLS)
            velocities.plot_velocities(COORS_FILE, SHP_FILE, stations_velos, SDs)
        # single station mode
        else:
            station = self.sel_station
            stations_velos, SDs, residuals = velocities.get_final_velocities(DST_FILE, EXCL_FILE, TS_DIR, plot_each_fit=True, 
                                            plot_each_outliers=False, period_year=period_year, stat=station, outlier_trash=outlier_trash,
                                            weeks_tresh_part=sub_tresh_weeks, weeks_tresh_all=full_tresh_weeks,
                                            plot_each_res=True, use_weights=WLS)

            if len(stations_velos) == 0:
                self.text_window.insert(END, 'stanica {} bola preskocena pre nedostatok dat\n'.format(station))
                self.text_window.insert(END, '**********************************************\n')
            else:
                vels_show = []
                SDs_show = []
                for label in ['n', 'e', 'u']:
                    vels_show.append(round(stations_velos[station][label], 5))
                    SDs_show.append(round(SDs[station][label], 6))

                self.text_window.insert(END, '{} velocities and sigmas [m/year]\n'.format(station))
                self.text_window.insert(END, 'v_n: {}, v_e: {}, v_u: {}, \n'.format(vels_show[0], vels_show[1], vels_show[2]))
                self.text_window.insert(END, 'Sv_n: {}, Sv_e: {}, Sv_u: {}, \n'.format(SDs_show[0], SDs_show[1], SDs_show[2]))
                self.text_window.insert(END, '**********************************************\n')

        # save residuals to csv
        if len(residuals) > 0 and len(save_path) > 0:
            df_res_all = pd.concat(list(residuals.values()))
            df_res_all.to_csv(save_path, float_format='%.8f')
        
           

def main():
  
    # root.geometry("250x150")
    app = TSPRO()
    app.mainloop()

if __name__ == '__main__':
    main()
