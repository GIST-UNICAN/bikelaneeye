# -*- coding: utf-8 -*-
"""Countspedestrians, bikers, and mobility scooter users that
cross user-defined virtual detectors in a video file. 
"""

from itertools import count, repeat
from argparse import ArgumentParser
from ultralytics import checks, YOLO
from pandas import DataFrame, concat, ExcelWriter
from shapely.geometry import LineString, Point
from multiprocessing import Manager, Pipe
from concurrent.futures import ProcessPoolExecutor
from logging import basicConfig, WARNING, INFO, info, exception, ERROR, warning
from time import monotonic
from collections import deque
from itertools import islice
import cv2
from math import atan2, pi
from pathlib import Path
import PySimpleGUI as sg
from random import choice
from sys import exit

debug_level = WARNING
basicConfig(
    filename='_bikelaneeye.log',
    level=debug_level,
    filemode='w',
    format='[%(levelname)s, %(asctime)s, %(threadName)s]: %(message)s')
det_file_names = ['clase',
                 'frame',
                 'id',
                 'top_x',
                 'top_y',
                 'w',
                 'h',
                 'conf']
q_unicode = ord("q")
spc_unicode = ord(" ")
minus_unicode = ord("-")
plus_unicode = ord("+")
comma_unicode = ord(",")
dot_unicode = ord(".")
z_unicode = ord("z")
x_unicode = ord("x")
t_unicode = ord("t")
unhandled_exception_text = "Unhandled exception occurred at %s."
sample_videos = ["sample_1.mp4", "sample_2.mp4"]

vertical_downwards = -pi/2
vertical_upwards = pi/2
csv_encoding = 'utf-8'

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,0,255)
thickness = 2
lineType = 2
line_color = (36,255,12)
temp_line_color = (255, 165, 0)
line_thickness = 2
baseconf = 0.01
wait_t_btwn_frames = 50
min_angle = 20 * pi / 180
manual_advance = None
old_files_suffix = "_old"
ext_excel = ".xlsx"
aggregation = 10 # seconds.
aggr_info_sheet_suffix = "_agr"


class DrawLineWidget(object):
    def __init__(self,
                 video_path,
                 counters,
                 counts,
                 cossings_queue,
                 t_0,
                 manual_advance = None,
                 wait_t_btwn_frames=wait_t_btwn_frames):
        self.video_path = str(video_path)
        self.counters=counters
        self.image_coordinates = deque()
        self.temp_line = []
        self.play = True
        self.wait_t_btwn_frames = wait_t_btwn_frames
        self.counts = counts
        self.cossings_queue = cossings_queue
        self.manual_advance = manual_advance
        self.t_0 = t_0

    def __call__(self):
        """Draws a counter line. Counts are shown as a tuple, where the first
        and second numbers show the e-scooter counts in the directions that 
        cross the counter line with an angle less and more than 180º. Likewise,
        the other four numbers represent people and bikes."""
        cv2.namedWindow('video', cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback('video', self.extract_coordinates)
        capture = cv2.VideoCapture(self.video_path)
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.duration = round(self.frame_count / self.fps)
        if not self.manual_advance:
            self.manual_advance = self.fps
        while capture.isOpened():
            if self.play:
                ret, self.frame = capture.read()
                if ret:
                    for line_name, coordinates in self.counters.items():
                        cv2.line(
                            self.frame,
                            tuple(islice(coordinates, None, 2)),
                            tuple(islice(coordinates, 2, None)),
                            line_color,
                            line_thickness)
                        if self.counts[line_name].done():
                            if (row_n := self.counts[line_name].result()["frame"].searchsorted(
                                round(capture.get(cv2.CAP_PROP_POS_FRAMES)), "left")) > 0:
                                cv2.putText(
                                    self.frame,
                                    f'{tuple(map(round, self.counts[line_name].result().loc[row_n-1, ["scooters_dir_0", "scooters_dir_1", "people_dir_0", "people_dir_1", "bicycle_dir_0", "bicycle_dir_1"]].to_list()))}',
                                    (round((coordinates[0]+coordinates[2])/2),
                                      round((coordinates[1]+coordinates[3])/2)),
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
                    if self.temp_line:
                        cv2.line(
                            self.frame,
                            self.temp_line[:2],
                            self.temp_line[2:],
                            temp_line_color,
                            line_thickness)
                    cv2.imshow("video", self.frame)
                else:
                    info(f'Playing {self.video_path} again from start.')
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            else:
                if self.temp_line:
                    frame_clone = self.frame.copy()
                    cv2.line(
                        frame_clone,
                        self.temp_line[:2],
                        self.temp_line[2:],
                        temp_line_color,
                        line_thickness)
                    cv2.imshow("video", frame_clone)
                else: 
                    cv2.imshow("video", self.frame)
            key = cv2.waitKey(self.wait_t_btwn_frames)
    
            # Close program with keyboard 'q'
            if key == q_unicode:
                cv2.destroyAllWindows()
                break
            elif key == spc_unicode:
                self.play = not self.play
            elif key == comma_unicode:
                capture.set(cv2.CAP_PROP_POS_FRAMES,
                            max(0,
                                round(capture.get(cv2.CAP_PROP_POS_FRAMES)
                                      - self.manual_advance)))
                continue
            elif key == z_unicode:
                capture.set(cv2.CAP_PROP_POS_FRAMES,
                            max(0,
                                round(capture.get(cv2.CAP_PROP_POS_FRAMES)
                                      - 10*self.manual_advance)))
                continue
            elif key == dot_unicode:
                if (
                    (new_frame := round(capture.get(cv2.CAP_PROP_POS_FRAMES)
                                        + self.manual_advance))
                    < self.frame_count):
                    
                    capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    continue
            elif key == x_unicode:
                if (
                    (new_frame := round(capture.get(cv2.CAP_PROP_POS_FRAMES)
                                        + 10*self.manual_advance))
                    < self.frame_count):
                    
                    capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    continue
            
        capture.release()
        cv2.destroyAllWindows()

    def extract_coordinates(self, event, x, y, flags, parameters):
        """Extracts coordinates from mouse events.
        
        :param event: The type of mouse event.
        :param x: The x-coordinate of the mouse event.
        :param y: The y-coordinate of the mouse event.
        :param flags: Additional flags for the mouse event.
        :param parameters: Additional parameters for the mouse event.
        """
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = deque((x,y))
            
        if event == cv2.EVENT_MOUSEMOVE and len(self.image_coordinates) == 2:
            self.temp_line = [*self.image_coordinates, x, y]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.temp_line.clear()
            if (
                    vertical_downwards
                    < atan2(
                        y - self.image_coordinates[1],
                        x - self.image_coordinates[0])
                    <= vertical_upwards):
                self.image_coordinates.extend((x,y))
            else:
                self.image_coordinates.extendleft((y,x))
            layout = [
                [sg.Text('Please enter a name for the counter or cancel.'),
                 sg.InputText(key="nombre")],
                [sg.Button('Ok'), sg.Button('Cancel')]]
            window = sg.Window('linea ok', layout)

            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
                    break
                if event=='Ok' and values['nombre']:
                    line_name=values['nombre']
                    self.counters[line_name] = (
                        self.image_coordinates)
                    self.counts[line_name] = executor.submit(
                        build_counts_df,
                        line_name,
                        self.image_coordinates,
                        full_df,
                        self.cossings_queue)
                    if not self.play:
                        cv2.line(
                            self.frame,
                            tuple(islice(self.image_coordinates, None, 2)),
                            tuple(islice(self.image_coordinates, 2, None)),
                            line_color,
                            line_thickness)            
                    break
            window.close()
                        
        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN and self.counters:
            líneas = tuple(self.counters.keys())
            layout = [
                [sg.Text('Please select the counter to erase.')],
                [sg.Combo(
                    values=líneas,
                    default_value=líneas[-1],
                    readonly=True,
                    key='selection')],
                [sg.Button('Ok'), sg.Button('Cancel')]]
            window = sg.Window('delete counter', layout)
            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
                    break
                if event=='Ok':
                    del self.counters[values["selection"]]
                    del self.counts[values["selection"]]
                    self.cossings_queue.put(values["selection"])
                    break
            window.close()

def get_counting_events(
        counter_name,
        counter_line_tpl,
        trackings,
        min_incid_angle,
        cls_to_idx={0:0, 1:2, 2:4}
        ):
    """
    Generates counting events based on the intersection of segments with a counting line and an incidence angle.

    This function iterates through tracked objects and detects when they cross a defined line within a specified range of angles, indicating a counting event. Each event is yielded with detailed information for further processing.

    Args:
        counter_name (str): The name of the counter used to identify the set of events.
        counter_line_tpl (tuple): Starting and ending coordinates of the counting line (x1, y1, x2, y2).
        trackings (DataFrame): A pandas DataFrame containing the tracks of objects to be counted.
        min_incid_angle (float): The minimum incidence angle required to count a crossing.
        cls_to_idx (dict, optional): A dictionary mapping object classes to specific indices. Defaults to {0: 0, 1: 2, 2: 4}.

    Yields:
        tuple: A tuple containing another tuple with frame information and detections, and a second one with tracking details.

    """

    min_angle = min_incid_angle
    max_angle = pi - min_incid_angle
    line_counter = LineString((
        (counter_line_tpl[0], counter_line_tpl[1]),
        (counter_line_tpl[2], counter_line_tpl[3])))
    yield (0, 0, 0, 0, 0, 0, 0), ()
    for obj, tracking in trackings.groupby("id"):
        detections = [0, 0, 0, 0, 0, 0]
        obj_detect_0 = obj_detect_1 = False
        for tracking_tpl in tracking.itertuples():
            if (
                    not (intersection_point := line_counter.intersection(
                            tracking_tpl.segment)).is_empty
                    and
                    min_angle <= abs((crossing_angle := atan2(
                        (
                            trayect_x := tracking_tpl.segment.coords[1][0]
                            - tracking_tpl.segment.coords[0][0])
                        * (
                            cont_y := counter_line_tpl[1]
                            - counter_line_tpl[3])
                        - (
                            trayect_y := tracking_tpl.segment.coords[1][1]
                           - tracking_tpl.segment.coords[0][1])
                        * (
                            cont_x := counter_line_tpl[0]
                            - counter_line_tpl[2]),
                        trayect_x * cont_x + trayect_y * cont_y))
                        ) <= max_angle):
                if crossing_angle > 0:
                    detections[cls_to_idx[tracking_tpl.clase]] = 1
                    obj_detect_0 = True
                else:
                    detections[cls_to_idx[tracking_tpl.clase] + 1] = 1
                    obj_detect_1 = True
                frame = (
                    tracking_tpl.frame_0
                    + tracking_tpl.pixels_to_frames 
                      * tracking_tpl.segment.project(intersection_point))
                yield (
                    (frame, *detections),
                    (
                        obj,
                        tracking_tpl.clase,
                        frame,
                        counter_name))
                if obj_detect_0 and obj_detect_1:
                    break

def build_counts_df(
        counter_name,
        line_counter,
        detections,
        cossings_queue):
    """Builds a DataFrame with counts of different objects detected in each frame.
    
    Parameters:
    - counter_name (str): The name of the counter.
    - line_counter (tuple): The line coordinates of the counter.
    - detections (list): List of detections.
    - cossings_queue (Queue): Queue to store crossing events.
    
    Returns:
    - counts_df (DataFrame): DataFrame with counts of different objects in each frame.
    
    Example usage:
    ```python
    counter_name = "Counter 1"
    line_counter = (0, 0, 100, 100)
    detections = [...]
    cossings_queue = Queue()
    
    counts_df = build_counts_df(counter_name, line_counter, detections, cossings_queue)
    print(counts_df)
    ```
    """
    counts_list, crossings_list = map(
        lambda f, rows: f(rows),
        (sorted, list),
        zip(*get_counting_events(
            counter_name,
            line_counter,
            detections,
            min_angle)))
    cossings_queue.put(crossings_list[1:])
    counts_df = DataFrame(
        counts_list,
        columns=(
            "frame",
            "scooters_dir_0",
            "scooters_dir_1",
            "people_dir_0",
            "people_dir_1",
            "bicycle_dir_0",
            "bicycle_dir_1"))
    counts_df.loc[:, "scooters_dir_0":"bicycle_dir_1"] = (
        counts_df.loc[:, "scooters_dir_0":"bicycle_dir_1"].cumsum())
    return counts_df

def is_not_none(x):
  return x is not None

def get_detect_rows():
  """Frame-by-frame this function returns the detection data of each object."""
  frames = count(1)
  for boxes in (result.boxes for result in tracking):
    frame = next(frames)
    # print(f"{frame=}")
    if all(map(is_not_none, (boxes.cls,
            boxes.id,
            boxes.conf,
            boxes.xywhn))):
      yield from zip(map(int, boxes.cls),
                     repeat(frame),
                     map(int, boxes.id),
                     # *zip(*boxes.xywhn),
                     *(tuple(map(float, t)) for t in zip(*boxes.xywh)),
                     map(float, boxes.conf))

def manage_crossings_df(
        cossings_queue,
        crossings_emitter_tube,
        columns = ("object", "clase", "frame", "contador"),
        dtype = "int64"):
    """Stores when each object crosses each counter line, sending the results
    as a dataframe through a pipe when the counting is finished."""
    try:
        cruces_df = DataFrame(
            columns = columns)#,
            # dtype = dtype)
        while True:
            message = cossings_queue.get()
            if isinstance(message, list):
                new_df = DataFrame(
                    message,
                    columns = columns)#,
                    # dtype = dtype)
                cruces_df = concat((cruces_df, new_df))
            elif isinstance(message, str):
                cruces_df.drop(
                    cruces_df[cruces_df["contador"] == message].index,
                    inplace=True)
            else:
                cruces_df.sort_values(['object','frame'],inplace=True)
                crossings_emitter_tube.send(cruces_df)
                break
    except:
        exception(
            unhandled_exception_text,
            manage_crossings_df.__name__)

def save_video_results(
        counts_path,
        counts,
        cossings_queue,
        fps,
        duration,
        crossings_receiver_tube,
        t_0):
    """This function saves the results of a video analysis to a file in Excel format.
    
    Parameters:
    - counts_path (str): The path to the file where the counts will be saved.
    - counts (dict): A dictionary containing the counts to be saved. The keys are the names of the counters and the values are the count results.
    - cossings_queue (Queue): A queue used for inter-process communication.
    - fps (float): The frames per second of the video.
    - duration (float): The duration of the video in seconds.
    - crossings_receiver_tube (Pipe): A pipe used for inter-process communication.
    - t_0 (float): The starting time of the video.
    
    Returns:
    - None
    
    Note:
    - This function uses the `csv` module to save the counters to a CSV file.
    - If the `counts_path` file already exists, it will be renamed with a suffix before saving the counts to a new file.
    - Counts are saved to separate sheets in an Excel file using the `pandas` library.
    - Counts are converted to time-based data by dividing the frame numbers by the frames per second.
    - The aggregated counts are calculated and saved to a separate sheet in the Excel file.
    - The crossings data is received through the `crossings_receiver_tube` pipe and saved to a separate sheet in the Excel file.
    """
    if counts_path.exists():
        counts_path.rename(counts_path.with_name(
            old_files_suffix.join((counts_path.stem, ext_excel))))
        info(
            "Pre-existing counts file renamed."
            )
    names_nd_counts = tuple(
        (counter_name, counter_counts.result())
        for counter_name, counter_counts in counts.items())
    cossings_queue.put(None)
    with ExcelWriter(counts_path) as excel_writer:
        for counter_name, counts_df in names_nd_counts:
            counts_df.rename(
                columns={"frame": "time (s)"},
                inplace=True)
            counts_df["time (s)"] /= fps
            counts_df.to_excel(
                excel_writer,
                sheet_name=counter_name,
                index=False)
            df_agregado = DataFrame(list(yield_aggregated(aggregation,
                                                            duration,
                                                            counts_df,
                                                            t_0)))
            df_agregado.iloc[1:, 1:] = df_agregado.diff().iloc[1:, 1:]
            df_agregado.to_excel(
                excel_writer,
                sheet_name=counter_name+aggr_info_sheet_suffix,
                index=False)
        cruces_df = crossings_receiver_tube.recv()
        cruces_df.rename(
            columns={"frame": "time (s)"},
            inplace=True)
        cruces_df["time (s)"] /= fps
        cruces_df.to_excel(
            excel_writer,
            sheet_name="cruces",
            index=False)
        
def yield_aggregated(aggregation, duration, counts_df, t_0):
    """This function returns aggregated values from a DataFrame based on the given aggregation, duration, DataFrame of counts, and initial time.
    
    Parameters:
    - aggregation (int): The aggregation interval in seconds.
    - duration (int): The duration in seconds.
    - counts_df (pandas.DataFrame): The DataFrame containing the counts.
    - t_0 (int): The initial time in seconds.
    
    Returns:
    - generator: A generator that yields the aggregated values.
    
    Example:
        >>> df = pd.DataFrame({'time (s)': [0, 1, 2, 3, 4, 5], 'count': [10, 20, 30, 40, 50, 60]})
        >>> for value in yield_aggregated(2, 5, df, 0):
        ...     print(value)
        [10 0]
        [30 2]
        [50 4]
    """
    n_fila_max = counts_df.shape[0] - 1
    for t in range(aggregation, duration+aggregation, aggregation):
        n_fila = min(
            n_fila_max,
            counts_df["time (s)"].searchsorted(t, "left") - 1)
        fila = counts_df.loc[n_fila]
        fila["time (s)"] = t - aggregation + t_0
        yield fila.values

if __name__ == "__main__":
    parser = ArgumentParser(
    add_help=True,
    prog="BikeLaneEye",
    description='Counts pedestrians, bikers, and mobility scooter users that'
                'cross user-defined virtual detectors in a video file. '
                'The "--hours", "--minutes" and "--seconds" flags can be used '
                'to specify the time when the video starts.')
    parser.add_argument("-f", "--file", type=str, default=None)
    parser.add_argument("-o", "--hours", type=int, default=0)
    parser.add_argument("-m", "--minutes", type=int, default=0)
    parser.add_argument("-s", "--seconds", type=int, default=0)
    parser.add_argument("-e", "--example",
                        action="store_true",
                        help='Randomly chooses on of the two available sample '
                        'videos to process.')
    args = parser.parse_args()
    if args.example:
        if args.file is None:
            args.file = choice(sample_videos)
            info(f'{args.file} will be analysed.')
        else:
            warning(f'Ignoring the "--example" flag.')
    elif args.file is None:
        print(
            'One of a file to analyse ("--file") or the "--example" flag is '
            'required. Use "-h" or "--help" for more information.')
        exit()
    video_path = Path(args.file)
    checks()
    model = YOLO('./best.pt')
    model.fuse()
    model.info(verbose=True)
    tracking = model.track(source=video_path, conf=baseconf)
    full_df = DataFrame.from_records(list(get_detect_rows()),
                                         columns = det_file_names)
    full_df.sort_values(['id','frame'], inplace=True)
    full_df.reset_index(inplace=True,drop=True)
    full_df['point'] = full_df.apply(
        lambda row: Point(row.top_x + row.w /2, row.top_y + row.h/2),
        axis=1)
    full_df.drop(['top_x','top_y','w','h'], axis=1, inplace=True)
    initial_frame_series = full_df.groupby('id')['frame'].shift()
    initial_frame_series.dropna(inplace=True)
    full_df['point_prev']=full_df['point'].shift()
    full_df = full_df.join(
        initial_frame_series,
        how="inner",
        rsuffix="_0")
    del initial_frame_series
    full_df["segment"] = full_df.apply(
        lambda row: LineString((row.point_prev, row.point)),
        axis=1)
    full_df = full_df.loc[
        full_df["segment"].apply(lambda s: s.length > 0)]
    full_df["frame"] = full_df.apply(
        (lambda row: (row.frame - row.frame_0) / row.segment.length),
        axis=1)
    full_df.rename(
        columns={"frame": "pixels_to_frames"},
        inplace=True)
    full_df.drop(["point", "point_prev"], axis=1, inplace=True)
    with ProcessPoolExecutor() as executor, Manager() as manager:
        cossings_queue = manager.Queue()
        crossings_receiver_tube, crossings_emitter_tube = Pipe(False)
        executor.submit(
            manage_crossings_df,
            cossings_queue,
            crossings_emitter_tube)
        counters = {}
        counts = {}
        t_0 = (
            3600 * args.hours +
            60 * args.minutes +
            args.seconds)
        line_updater = DrawLineWidget(
            video_path,
            counters,
            counts,
            cossings_queue,
            t_0)
        line_updater()
        counts_path = video_path.with_suffix(ext_excel)
        save_video_results(
            counts_path,
            counts,
            cossings_queue,
            line_updater.fps,
            line_updater.duration,
            crossings_receiver_tube,
            line_updater.t_0)