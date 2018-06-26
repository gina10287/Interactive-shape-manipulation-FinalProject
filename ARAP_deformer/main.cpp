#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <limits>
#include <Windows.h>
#include <gl/GL.h>
#include <glut.h>
#include <time.h>

#include "glm.h"
#include "mtxlib.h"
#include "trackball.h"
#include "LeastSquaresSparseSolver.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <string> 
#include <algorithm>

using namespace std;
using namespace Eigen;

// global variables
_GLMmodel *mesh;
_GLMmodel *mesh_original;
TimeRecorder time_record;
char filename[10];
int WindWidth = 800, WindHeight = 800;

// --------------------------------------------------- global parameters -----------------------------------------------------------
// ---------------------------------------------
int last_x , last_y;
int select_x, select_y;
int mouse_x, mouse_y, track_vertex = 0;
float longestEdge = 0;

float dis_smallest, dis_limit = 0.00003;
int num_vertices, num_triangles;
int handle_size = 0;
int num_UncurrentHandle = 0, num_SilIdx = 0, num_SilVertex = 0, num_Silhouette = 0;
int contourTriangle_size = 0;
int num_feature = 0;

int num_cp = 500;
int iteration_limit = 5;
int iteration_times = 0;
int selected_handle_id = -1;

float eyex = 0.0;
float eyey = 0.0;
float eyez = 3.0;
float depthOfSilhouette = 0.0f;
float lengthOfContour = 0.0f;

bool iteration = true;
bool draw_silhouette_flag = false;
bool draw_feature_flag = false;
bool deform_flag = false;
bool deform_feature_flag = false;
bool display_wire_flag = false;
typedef enum { CLEAR_MODE, SELECT_MODE, DRAW_MODE, FIND_CONTOUR, DEFORM_MODE1, DEFORM_MODE2, FEATURE_MODE } ControlMode;
ControlMode current_mode = SELECT_MODE;

// --------------------------------------------- Silhouette sketch parameters ------------------------------------------------------
// ---------------------------------------------
struct Edge
{
	int v1;
	int v2;
	float w1;
	float w2;
	float length_dif;
	vector3 p1;
	vector3 p2;
	vector3 target;
};
typedef struct Edge Edge;
// ---------------------------------------------
struct Position
{
	float x;
	float y;
	float z;
};
typedef struct Position Position;
// ---------------------------------------------
struct Cotangent
{
	int idx;
	float weight;
	vector<float> cotangent;
};
typedef struct Cotangent Cotangent;
// ---------------------------------------------

vector<float*> colors;

vector<vector <int>> handles;

vector<vector <int>> connectivity;
vector<vector3> delta_original;
vector<vector3> delta;
vector<vector <Cotangent>> cot_connectivity_original;
vector<vector <Cotangent>> cot_connectivity;
vector<vector3> cot_delta_original;
vector<vector3> cot_delta_update;
vector<vector3> cot_delta;

vector<vector <Position>> E_original;
vector<vector <Position>> E_new;

vector<vector <int>> vertexFindex; //store which triangles are vertex in
vector<vector <float>> triangles_position; //center of all triangles in mesh
vector<vector <float>> resultList;

vector<vector <int>> handles_triangles; //store the triangles in selected handles
vector<int> handles_vertex; //store the vertices in selected handle

vector<int> vertex_value, vertex_value_ori; //value of vertices in current handle, is used to split he vertex into + and - groups
vector<int> handles_contour_triangles; //triangles in current handle that will contains contour
vector<int> contour_triangles; //same as handles_contour_triangles, but is used to avoid finding a repeat triangles
vector<float> contour_normal; //store the normalized value correspond to contour vertex 
vector2 refind(0, 0);

vector<vector3> mouse_tracking; //stores the currently draw silhouette vertices
vector<vector3> mouse_tracking_original; //just for draw
vector<float> silhouette_normal; //store the normalized value correspond to silhouette vertex

vector<Edge> contour_edge;
vector<Edge> first_contour_edge;
vector<Edge> reverse_contour_edge;

vector<int> co_silVertex; //stores the idx vertices not in contour vertex
vector<Edge> co_silhouette; //stores the silhouette vertices correspond to contour vertex
vector<int> SilIdx;
vector<vector <int>> SilVertex;
vector<vector <Edge>> Silhouette;
vector <int> allVertex_notInHandle;

// ---------------------------------------------- Feature sketch parameters --------------------------------------------------------
// ---------------------------------------------
struct Feature
{
	int trackingIdx;
	vector<int> trackingPath; //vertex when go to next nearestIdx will pass

	int meshIdx;
	vector<int> meshPath; //vertex when go to next nearestIdx will pass
};
typedef struct Feature Feature;

// ---------------------------------------------
vector<Feature> FeatureTrack;

vector<vector3> feature_tracking; //stores the currently draw feature sketching vertices
vector<float> feature_depth; //stores the correspond depth of "feature_tracking"
vector<int> nearestIdx;
vector<int> randomCpList;

vector<vector <int>> FeaturePath; //stores the vertex in order that this feature sketching will pass
vector<int> feature_path; //stores the vertex in order that this feature sketching will pass
vector<vector3> correspond_sketch;

vector<int> feature_path_vertices;
vector<int> feature_roundVertices;


// -------------------------------------------------- Initial functions ------------------------------------------------------------
// ---------------------------------------------
// global functions
void ini_resize(void);
void findLongestLength(void);
bool storeOrNot(vector<int> *vec, int connected_idx);
void get_Connectivity(void);
void get_Delta(void);
void getUmbrella(void);
// ---------------------------------------------
void get_E(void);
void update_E(void);
Matrix3f get_R(int vertex_i);
void get_triangles_position(void);
// ---------------------------------------------
void LeastSquareSolver_cot(void);
float getCot_Value(int pSrc, int p0, int p1);
int getConnectivity_j(int i, int j);
void getCot_Connectivity(void);
void getCot_Delta(void);
void getContangent(void);

// --------------------------------------------- Silhouette sketch functions -------------------------------------------------------
// ---------------------------------------------
void unHandleVertex(void);
void getSolverIdx(void);
void solverTest(void);
void getOriginalDelta(void);
void updateDelta(void);
void LeastSquareSolver_v2(void);
void LeastSquareSolver(void);
void Reconstruction(void);
void BackToOriginal(void);
// ---------------------------------------------
void zeroTriangles(void);
void getHandlesTriangles(void);
// ---------------------------------------------
void inSilhouetteOrNot(void);
void getContourLength(void);
void getSilhouetteLength(void);
void findCorrespondence(void);
void getCorrespondence(void);
void drawCorrespondence(void);
// ---------------------------------------------
float get_depth(vector2 _2Dpos);
void regetSilhouetteDepth(void);
void getSilhouetteDepth(void);
void DrawFirstSilhouette(void);
void DrawSilhouette(void);
void DrawSilhouette_test(void);
// ---------------------------------------------
vector3 getPosition(int idx);
Edge getEdge(int v1, int v2);
int findAotherTriangle(int idx_tri, int v1, int v2);
void findEdge(int mode, int idx_tri, int v1, int v2);
void first_findEdge(int startPoint);
void reverse_findEdge(int startPoint);
void combine_Edge(void);
void getWeightOfEdge(void);
void getLengthOfEdge(void);
void findContour(void);
void firstFindContour(int startPoint);
void reFindContour(int startPoint);
void drawContour(void);

// ----------------------------------------------- Feature sketch functions --------------------------------------------------------
// ---------------------------------------------
int getPathIdx(int idx);
void LeastSquareSolver_update(void);
void scaleCotDelta(void);
void updateCotangent(vector<int> *);
void LeastSquareSolver_feature(void);
void setFeaturePosition(void);
void get_RandomCp(void);
void getVerticesInPathTriangles(void);
// ---------------------------------------------
void testFeatureSize(void);
vector3 getTrackingPosition(int i, int j);
void SketchCorrespondence();
int walkEdge(int a, int b);
int findPath(int index, int idx_0, int idx_1);
void featureTracking(void);
void findAllNearestVertex(void);
void getFeatureSketch(void);
void DrawFeaturePath(void);
void DrawFeature(void);

// -------------------------------------------------- Display functions ------------------------------------------------------------
// ---------------------------------------------
void Reshape(int width, int height);
void Display(void);
// ---------------------------------------------
vector3 Unprojection_silhouettes(vector2 _2Dpos, float depth);
vector3 Unprojection(vector2 _2Dpos);
vector2 projection_helper(vector3 _3Dpos);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void keyboard(unsigned char key, int x, int y);
void timf(int value);


/*--------------------------------------------- UI -------------------------------------------------------*/
// ---------------------------------------------
int winw = 250;
int winh = 800;

// Mouse Stuff -----------------------------------------------------
struct Mouse
{
	int x;		/*	the x coordinate of the mouse cursor	*/
	int y;		/*	the y coordinate of the mouse cursor	*/
	int lmb;	/*	is the left button pressed?		*/
	int mmb;	/*	is the middle button pressed?	*/
	int rmb;	/*	is the right button pressed?	*/
	int xpress; /*	stores the x-coord of when the first button press occurred	*/
	int ypress; /*	stores the y-coord of when the first button press occurred	*/
};
typedef struct Mouse Mouse;
Mouse TheMouse = { 0, 0, 0, 0, 0 };

// Button Stuff ----------------------------------------------------
typedef void(*ButtonCallback)();
struct Button
{
	int   x;							/* top left x coord of the button */
	int   y;							/* top left y coord of the button */
	int   w;							/* the width of the button */
	int   h;							/* the height of the button */
	int	  state;						/* the state, 1 if pressed, 0 otherwise */
	int	  highlighted;					/* is the mouse cursor over the control? */
	char* label;						/* the text label of the button */
	ButtonCallback callbackFunction;	/* A pointer to a function to call if the button is pressed */
};
typedef struct Button Button;


// Call back functions
void ClearCallback(void);
void SelectHandleCallback(void);
void ContourCallback(void);
void SilhouetteCallback(void);
void Deform0Callback(void);
void Deform1Callback(void);
void FeatureCallback(void);
void DisplayCallback(void);

// MyButton Btn_SelectHandle Btn_Contour Btn_AnotherContour Btn_Silhouette Btn_Deform
int positionX = 30, positionY = 85, diff = 60, btn_width = 200, btn_height = 30;
Button Btn_Clear = { positionX, positionY - 1 * diff, btn_width, btn_height, 0, 0, "Clear", ClearCallback };
Button Btn_Display = { positionX, positionY, btn_width, btn_height, 0, 0, "Hide Wire", DisplayCallback };
Button Btn_SelectHandle = { positionX, positionY + 1 * diff, btn_width, btn_height, 0, 0, "Select Handle", SelectHandleCallback };
Button Btn_Contour = { positionX, positionY + 2 * diff, btn_width, btn_height, 0, 0, "Find Contour", ContourCallback };
Button Btn_Silhouette = { positionX, positionY + 3 * diff, btn_width, btn_height, 0, 0, "Draw Silhouette", SilhouetteCallback };
Button Btn_Deform0 = { positionX, positionY + 7.5 * diff, btn_width, btn_height, 0, 0, "Deform Mode 0", Deform0Callback };
Button Btn_Deform1 = { positionX, positionY + 8.5 * diff, btn_width, btn_height, 0, 0, "Deform Mode 1", Deform1Callback };
Button Btn_Feature = { positionX, positionY + 9.5 * diff, btn_width, btn_height, 0, 0, "Feature Sketch", FeatureCallback };

int degreeX = positionX + 25 + 3, degree_width = btn_width - 2 * 25 - 6, n_diff = 60;
int de_x = positionX, de_y = positionY + 4.5 * diff, in_x = positionX + btn_width - 25;
int name_width = 210;

struct WeightBtn{
	int weight;
	Button name;
	Button decrease;
	Button increase;
	Button degree;
};
typedef struct WeightBtn WeightBtn;

WeightBtn Weight_unCurrentHandle;
WeightBtn Weight_targetOri;
WeightBtn Weight_target;
WeightBtn Weight_feature;

// Callback functions and related functions ------------------------
void updateMode() {
	if (current_mode == FEATURE_MODE)
		updateCotangent(&feature_roundVertices);
}

void ClearCallback() {
	updateMode();
	current_mode = CLEAR_MODE;
}
void SelectHandleCallback() {
	updateMode();
	current_mode = SELECT_MODE;
}
void ContourCallback() {
	updateMode();
	current_mode = FIND_CONTOUR;
}
void SilhouetteCallback() {
	updateMode();
	current_mode = DRAW_MODE;
}
void Deform0Callback() {
	updateMode();
	current_mode = DEFORM_MODE1;
}
void Deform1Callback() {
	updateMode();
	current_mode = DEFORM_MODE2;
}
void FeatureCallback() {
	updateMode();
	current_mode = FEATURE_MODE;
}
void DisplayCallback() {
	if (!display_wire_flag) {
		display_wire_flag = true;
		Btn_Display.label = "Hide Wire";
	}
	else {
		display_wire_flag = false;
		Btn_Display.label = "Show Wire";
	}
}

void decrese_obj(WeightBtn *obj) {
	if (obj->weight > 0) {
		obj->weight -= 1;
	}
}
void increse_obj(WeightBtn *obj) {
	if (obj->weight < 10) {
		obj->weight += 1;
	}
}

void weight1Callback_decrease() {
	decrese_obj(&Weight_unCurrentHandle);
}
void weight1Callback_increase() {
	increse_obj(&Weight_unCurrentHandle);
}
void weight2Callback_decrease() {
	decrese_obj(&Weight_targetOri);
}
void weight2Callback_increase() {
	increse_obj(&Weight_targetOri);
}
void weight3Callback_decrease() {
	decrese_obj(&Weight_target);
}
void weight3Callback_increase() {
	increse_obj(&Weight_target);
}

void weight4Callback_decrease() {
	if (Weight_feature.weight >= -90) {
		Weight_feature.weight -= 10;
		deform_feature_flag = true;
	}
}
void weight4Callback_increase() {
	if (Weight_feature.weight <= 90) {
		Weight_feature.weight += 10;
		deform_feature_flag = true;
	}
}

// UI functions ----------------------------------------------------
void Font(void *font, char *text, int x, int y)
{
	glRasterPos2i(x, y);

	while (*text != '\0')
	{
		glutBitmapCharacter(font, *text);
		++text;
	}
}
int ButtonClickTest(Button* b, int x, int y)
{
	if (b)
	{
		if (x > b->x      &&
			x < b->x + b->w &&
			y > b->y      &&
			y < b->y + b->h) {
			return 1;
		}
	}
	return 0;
}
void ButtonRelease(Button *b, int x, int y)
{
	if (b)
	{
		/* If the mouse button was pressed within the button area as well as being released on the button..... */
		if (ButtonClickTest(b, x, y))
		{
			/* Then if a callback function has been set, call it. */
			if (handles.size()>0 && b->callbackFunction) {
				b->callbackFunction();
			}
			else if (b->h == 25 || b->y == Btn_Clear.y || b->y == Btn_Display.y || b->y == Btn_Feature.y) {
				b->callbackFunction();
			}
		}

		/* Set state back to zero. */
		b->state = 0;
	}
}
void ButtonPress(Button *b, int x, int y)
{
	if (b)
	{
		/* if the mouse click was within the buttons client area, set the state to true. */
		if (ButtonClickTest(b, x, y) && handles.size() > 0)
		{
			if (b->label == "Clear" || b->label == "Select Handle" || b->label == "Find Contour" || b->label == "Feature Sketch") {
				mouse_tracking.clear();
				mouse_tracking_original.clear();
			}
			b->state = 1;
		}
	}
}
void ButtonPassive(Button *b, int x, int y)
{
	if (b)
	{
		/* if the mouse moved over the control */
		if (ButtonClickTest(b, x, y))
		{
			/* If the cursor has just arrived over the control, set the highlighted flag and force a redraw.
			*	The screen will not be redrawn again until the mouse is no longer over this control */
			if (b->highlighted == 0) {
				b->highlighted = 1;
				glutPostRedisplay();
			}
		}
		else

			/* If the cursor is no longer over the control, then if the control is highlighted (ie, the mouse has
			* JUST moved off the control) then we set the highlighting back to false, and force a redraw. */

		if (b->highlighted == 1)
		{
			b->highlighted = 0;
			glutPostRedisplay();
		}
	}
}
char* intToChar(int degree) {
	float f_degree = (float)degree / 10;
	string str = to_string(f_degree);
	if (degree >= 0)
		str = str.substr(0,3);
	else
		str = str.substr(0, 4);

	char *cstr = new char[str.length() + 1];
	strcpy(cstr, str.c_str());
	return cstr;
}
void ButtonDraw(Button *b)
{
	int fontx;
	int fonty;

	if (b)
	{
		if (b->w == name_width) {
			/* Calculate the x and y coords for the text string in order to center it. */
			fontx = b->x + (b->w - glutBitmapLength(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)b->label)) / 2;
			fonty = b->y + (b->h + 10) / 2;
			
			glColor3f(0.2f, 0.6f, 0.6f);
			Font(GLUT_BITMAP_HELVETICA_18, b->label, fontx, fonty);
		}
		// degree of contour count
		else if (b->x == degreeX) {
			glColor3f(1, 1, 1);
			/* draw background for the button. */
			glBegin(GL_QUADS);
			glVertex2i(b->x, b->y);
			glVertex2i(b->x, b->y + b->h);
			glVertex2i(b->x + b->w, b->y + b->h);
			glVertex2i(b->x + b->w, b->y);
			glEnd();

			/* Calculate the x and y coords for the text string in order to center it. */
			fontx = b->x + (b->w - glutBitmapLength(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)b->label)) / 2;
			fonty = b->y + (b->h + 10) / 2;

			glColor3f(0, 0, 0);
			char *c_degree;
			if (b->y == de_y)
				c_degree = intToChar(Weight_unCurrentHandle.weight);
			else if (b->y == de_y + n_diff)
				c_degree = intToChar(Weight_targetOri.weight);
			else if (b->y == de_y + 2 * n_diff)
				c_degree = intToChar(Weight_target.weight);
			else {
				c_degree = intToChar(Weight_feature.weight);
			}
			Font(GLUT_BITMAP_HELVETICA_18, c_degree, fontx, fonty);
		}
		else {
			if (b->label == Btn_Clear.label) {
				if (b->highlighted)
					glColor3f(0.8f, 0.2f, 0.2f);
				else
					glColor3f(1.0f, 0.3f, 0.3f);
			}
			// increase & decrease
			else if (b->h == b->w) {
				/* We will indicate that the mouse cursor is over the button by changing its colour. */
				if (b->highlighted)
					glColor3f(0.7f, 0.7f, 0.8f);
				else
					glColor3f(0.6f, 0.6f, 0.6f);
			}
			else if (b->y == Btn_Display.y) {
				if (b->highlighted)
					glColor3f(0.5f, 0.1f, 0.5f);
				else
					glColor3f(0.7f, 0.2f, 0.7f);
			}
			else {
				/* We will indicate that the mouse cursor is over the button by changing its colour. */
				if (b->highlighted)
					glColor3f(0.0f, 0.4f, 0.4f);
				else
					glColor3f(0.1f, 0.6f, 0.6f);
			}
			/* draw background for the button. */
			glBegin(GL_QUADS);
			glVertex2i(b->x, b->y);
			glVertex2i(b->x, b->y + b->h);
			glVertex2i(b->x + b->w, b->y + b->h);
			glVertex2i(b->x + b->w, b->y);
			glEnd();

			/* Draw an outline around the button with width 3 */
			glLineWidth(2);

			/* The colours for the outline are reversed when the button. */
			if (b->state)
				glColor3f(0.4f, 0.4f, 0.4f);
			else
				glColor3f(0.8f, 0.8f, 0.8f);

			glBegin(GL_LINE_STRIP);
			glVertex2i(b->x + b->w, b->y);
			glVertex2i(b->x, b->y);
			glVertex2i(b->x, b->y + b->h);
			glEnd();

			if (b->state)
				glColor3f(0.8f, 0.8f, 0.8f);
			else
				glColor3f(0.4f, 0.4f, 0.4f);

			glBegin(GL_LINE_STRIP);
			glVertex2i(b->x, b->y + b->h);
			glVertex2i(b->x + b->w, b->y + b->h);
			glVertex2i(b->x + b->w, b->y);
			glEnd();

			glLineWidth(1);


			/* Calculate the x and y coords for the text string in order to center it. */
			fontx = b->x + (b->w - glutBitmapLength(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)b->label)) / 2;
			if (b->h == b->w)
				fonty = b->y + (b->h + 10) / 2;
			else
				fonty = b->y + (b->h + 10) / 2 + 5;

			/* if the button is pressed, make it look as though the string has been pushed
			*	down. It's just a visual thing to help with the overall look.... */
			if (b->state) {
				fontx += 2;
				fonty += 2;
			}

			/* If the cursor is currently over the button we offset the text string and draw a shadow */
			if (b->highlighted)
			{
				glColor3f(0, 0, 0);
				Font(GLUT_BITMAP_HELVETICA_18, b->label, fontx, fonty);
				fontx--;
				fonty--;
			}

			glColor3f(1, 1, 1);
			Font(GLUT_BITMAP_HELVETICA_18, b->label, fontx, fonty);

		}
		
	}
}
void Init()
{
	glEnable(GL_LIGHT0);
	//Weight_unCurrentHandle, Weight_targetOri, Weight_target
	Weight_unCurrentHandle.weight = 0;
	Weight_targetOri.weight = 0;
	Weight_target.weight = 5;
	Weight_feature.weight = 10;

	Weight_unCurrentHandle.decrease = { de_x, de_y, 25, 25, 0, 0, "-", weight1Callback_decrease };
	Weight_unCurrentHandle.increase = { in_x, de_y, 25, 25, 0, 0, "+", weight1Callback_increase };
	Weight_unCurrentHandle.name = { de_x - 5, de_y - 25, name_width, 25, 0, 0, "Un-current handle vertex", 0 };
	Weight_unCurrentHandle.degree = { degreeX, de_y, degree_width, 25, 0, 0, "0", 0 };
	
	Weight_targetOri.decrease = { de_x, de_y +  n_diff, 25, 25, 0, 0, "-", weight2Callback_decrease };
	Weight_targetOri.increase = { in_x, de_y +  n_diff, 25, 25, 0, 0, "+", weight2Callback_increase };
	Weight_targetOri.name = { de_x - 5, de_y +  n_diff - 25, name_width, 25, 0, 0, "Contour original position", 0 };
	Weight_targetOri.degree = { degreeX, de_y +  n_diff, degree_width, 25, 0, 0, "0", 0 };
	
	Weight_target.decrease = { de_x, de_y + 2 * n_diff, 25, 25, 0, 0, "-", weight3Callback_decrease };
	Weight_target.increase = { in_x, de_y + 2 * n_diff, 25, 25, 0, 0, "+", weight3Callback_increase };
	Weight_target.name = { de_x - 5, de_y + 2 * n_diff - 25, name_width, 25, 0, 0, "Contour target position", 0 };
	Weight_target.degree = { degreeX, de_y + 2 * n_diff, degree_width, 25, 0, 0, "0", 0 };

	Weight_feature.decrease = { de_x, de_y + 5.6 * diff, 25, 25, 0, 0, "-", weight4Callback_decrease };
	Weight_feature.increase = { in_x, de_y + 5.6 * diff, 25, 25, 0, 0, "+", weight4Callback_increase };
	Weight_feature.degree = { degreeX, de_y + 5.6 * diff, degree_width, 25, 0, 0, "0", 0 };
}
void Draw2D()
{
	ButtonDraw(&Btn_Clear);
	ButtonDraw(&Btn_SelectHandle);
	ButtonDraw(&Btn_Contour);
	ButtonDraw(&Btn_Silhouette);
	ButtonDraw(&Btn_Deform0);
	ButtonDraw(&Btn_Deform1);
	ButtonDraw(&Btn_Feature);
	ButtonDraw(&Btn_Display);
	

	//Weight_unCurrentHandle, Weight_targetOri, Weight_target
	ButtonDraw(&Weight_unCurrentHandle.name);
	ButtonDraw(&Weight_unCurrentHandle.degree);
	ButtonDraw(&Weight_unCurrentHandle.decrease);
	ButtonDraw(&Weight_unCurrentHandle.increase);
	
	ButtonDraw(&Weight_targetOri.name);
	ButtonDraw(&Weight_targetOri.degree);
	ButtonDraw(&Weight_targetOri.decrease);
	ButtonDraw(&Weight_targetOri.increase);
	
	ButtonDraw(&Weight_target.name);
	ButtonDraw(&Weight_target.degree);
	ButtonDraw(&Weight_target.decrease);
	ButtonDraw(&Weight_target.increase);

	ButtonDraw(&Weight_feature.degree);
	ButtonDraw(&Weight_feature.decrease);
	ButtonDraw(&Weight_feature.increase);
}
void Draw()
{
	/* Clear the background */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* Disable depth test and lighting for 2D elements */
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glPushMatrix();
	/* Set the orthographic viewing transformation */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, winw, winh, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPopMatrix();

	/* Draw the 2D overlay */
	Draw2D();

	/* Bring the back buffer to the front and vice-versa*/
	glutSwapBuffers();
}
void Resize(int w, int h)
{
	winw = w;
	winh = h;

	glViewport(0, 0, w, h);
}
void MouseButton(int button, int state, int x, int y)
{
	TheMouse.x = x;
	TheMouse.y = y;

	if (state == GLUT_DOWN)
	{
		if (!(TheMouse.lmb || TheMouse.mmb || TheMouse.rmb)) {
			TheMouse.xpress = x;
			TheMouse.ypress = y;
		}
		switch (button)
		{
		case GLUT_LEFT_BUTTON:
			TheMouse.lmb = 1; 
			ButtonPress(&Btn_Clear, x, y);
			ButtonPress(&Btn_SelectHandle, x, y);
			ButtonPress(&Btn_Contour, x, y);
			ButtonPress(&Btn_Silhouette, x, y);
			ButtonPress(&Btn_Deform0, x, y);
			ButtonPress(&Btn_Deform1, x, y);
			ButtonPress(&Btn_Feature, x, y);
			ButtonPress(&Btn_Display, x, y);

			ButtonPress(&Weight_unCurrentHandle.decrease, x, y);
			ButtonPress(&Weight_unCurrentHandle.increase, x, y);
			ButtonPress(&Weight_targetOri.decrease, x, y);
			ButtonPress(&Weight_targetOri.increase, x, y);
			ButtonPress(&Weight_target.decrease, x, y);
			ButtonPress(&Weight_target.increase, x, y);

			ButtonPress(&Weight_feature.decrease, x, y);
			ButtonPress(&Weight_feature.increase, x, y);
			break;
		case GLUT_MIDDLE_BUTTON:
			TheMouse.mmb = 1;
			break;
		case GLUT_RIGHT_BUTTON:
			TheMouse.rmb = 1;
			break;
		}
	}
	else
	{
		switch (button)
		{
		case GLUT_LEFT_BUTTON:
			TheMouse.lmb = 0;
			ButtonRelease(&Btn_Clear, x, y);
			ButtonRelease(&Btn_SelectHandle, x, y);
			ButtonRelease(&Btn_Contour, x, y);
			ButtonRelease(&Btn_Silhouette, x, y);
			ButtonRelease(&Btn_Deform0, x, y);
			ButtonRelease(&Btn_Deform1, x, y);
			ButtonRelease(&Btn_Feature, x, y);
			ButtonRelease(&Btn_Display, x, y);

			ButtonRelease(&Weight_unCurrentHandle.decrease, x, y);
			ButtonRelease(&Weight_unCurrentHandle.increase, x, y);
			ButtonRelease(&Weight_targetOri.decrease, x, y);
			ButtonRelease(&Weight_targetOri.increase, x, y);
			ButtonRelease(&Weight_target.decrease, x, y);
			ButtonRelease(&Weight_target.increase, x, y);

			ButtonRelease(&Weight_feature.decrease, x, y);
			ButtonRelease(&Weight_feature.increase, x, y);
			break;
		case GLUT_MIDDLE_BUTTON:
			TheMouse.mmb = 0;
			break;
		case GLUT_RIGHT_BUTTON:
			TheMouse.rmb = 0;
			break;
		}
	}
	glutPostRedisplay();
}
void ButtonPassive_all(int x, int y) {
	ButtonPassive(&Btn_Clear, x, y);
	ButtonPassive(&Btn_SelectHandle, x, y);
	ButtonPassive(&Btn_Contour, x, y);
	ButtonPassive(&Btn_Silhouette, x, y); 
	ButtonPassive(&Btn_Deform0, x, y);
	ButtonPassive(&Btn_Deform1, x, y);
	ButtonPassive(&Btn_Feature, x, y);
	ButtonPassive(&Btn_Display, x, y);

	ButtonPassive(&Weight_unCurrentHandle.decrease, x, y);
	ButtonPassive(&Weight_unCurrentHandle.increase, x, y);
	ButtonPassive(&Weight_targetOri.decrease, x, y);
	ButtonPassive(&Weight_targetOri.increase, x, y);
	ButtonPassive(&Weight_target.decrease, x, y);
	ButtonPassive(&Weight_target.increase, x, y);

	ButtonPassive(&Weight_feature.decrease, x, y);
	ButtonPassive(&Weight_feature.increase, x, y);
}
void MouseMotion(int x, int y)
{
	int dx = x - TheMouse.x;
	int dy = y - TheMouse.y;
	TheMouse.x = x;
	TheMouse.y = y;
	ButtonPassive_all(x, y);
	glutPostRedisplay();
}
void MousePassiveMotion(int x, int y)
{
	int dx = x - TheMouse.x;
	int dy = y - TheMouse.y;
	TheMouse.x = x;
	TheMouse.y = y;
	ButtonPassive_all(x, y);

}

// ---------------------------------------------
/*--------------------------------------------- UI Finish ------------------------------------------------*/


/*----------------------------------------------- ARAP ---------------------------------------------------*/
// ---------------------------------------------
// Initial functions when mesh loaded, only exceute once
//Initial and resize vectors
void ini_resize()
{
	time_record.ResetTimer();

	num_vertices = mesh->numvertices;
	num_triangles = mesh->numtriangles;

	connectivity.resize(num_vertices);
	cot_connectivity.resize(num_vertices);

	vertexFindex.resize(num_vertices);
	delta.resize(num_vertices);
	cot_delta.resize(num_vertices);

	resultList.resize(num_vertices, vector<float>(3, 0.0f));

	E_original.resize(num_vertices);
	E_new.resize(num_vertices);

	triangles_position.resize(num_triangles, vector<float>(3, 0.0f));
	vertex_value.resize(num_vertices, -100);
	vertex_value_ori.resize(num_vertices, -100);

	cout << "number of vertices = " << num_vertices << endl;
	cout << "number of triangles = " << num_triangles << endl << endl;
}

void findLongestLength() {

	int idx_1, idx_2;
	float length;
	for (int i = 0; i < num_triangles; i++) {

		for (int j = 0; j < 3; j++) {
			idx_1 = mesh->triangles[(i)].vindices[j] - 1;
			vector3 pt_1(mesh->vertices[3 * (idx_1 + 1) + 0], mesh->vertices[3 * (idx_1 + 1) + 1], mesh->vertices[3 * (idx_1 + 1) + 2]);

			for (int k = j + 1; k < 3; k++) {
				idx_2 = mesh->triangles[(i)].vindices[k] - 1;
				vector3 pt_2(mesh->vertices[3 * (idx_2 + 1) + 0], mesh->vertices[3 * (idx_2 + 1) + 1], mesh->vertices[3 * (idx_2 + 1) + 2]);
				length = (pt_2 - pt_1).lengthSqr();

				if (length > longestEdge) {
					longestEdge = length;
				}
			}
		}
	}
	//cout << "longestEdge= " << longestEdge << endl;
}

// ---------------------------------------------
bool storeOrNot(vector<int> *vec, int connected_idx) {
	bool store = true;
	for (int i = 0; i < vec->size(); i++) {
		if (vec->at(i) == connected_idx) {
			store = false;
		}
	}
	return store;
}

//Connectivity stores the connectivity(neighbor) of every vertex
//Like vertex-0 connect with vertex-2,vertex-3,vertex-4......, then Connectivity[0]=[2,3,4....]
void get_Connectivity() {

	int store_idx, connected_idx;
	for (int i = 0; i < num_triangles; i++) {

		for (int j = 0; j < 3; j++) {
			store_idx = mesh->triangles[(i)].vindices[j] - 1;

			//store triangle for each vertex
			if (storeOrNot(&vertexFindex[store_idx], i)) {
				vertexFindex[store_idx].push_back(i);
			}

			for (int k = 0; k < 3; k++) {
				connected_idx = mesh->triangles[(i)].vindices[k] - 1;
				//store connected vertex for each vertex
				//if (j != k && storeVertexOrNot(store_idx, connected_idx)) {
				if (j != k && storeOrNot(&connectivity[store_idx], connected_idx)) {
					connectivity[store_idx].push_back(connected_idx);
				}
			}

		}
	}

	for (int i = 0; i < num_vertices; i++) {
		cot_connectivity[i].resize(connectivity[i].size());
		E_original[i].resize(connectivity[i].size());
		E_new[i].resize(connectivity[i].size());
	}
}

//Get original delta =  (connectivity) * (vertex position)
void get_Delta() {

	float conn_size;
	for (int i = 0; i < num_vertices; i++) {

		vector3 buffer(0, 0, 0);
		conn_size = (-1 / (float)(connectivity[i].size()));

		for (int j = 0; j < connectivity[i].size(); j++) {
			for (int k = 0; k < 3; k++) {
				buffer[k] += mesh->vertices[3 * (connectivity[i][j] + 1) + k];
			}
		}

		for (int k = 0; k < 3; k++) {
			delta[i][k] = mesh->vertices[3 * (i + 1) + k] + conn_size*buffer[k];
		}

	}
}

void getUmbrella() {
	get_Connectivity();
	get_Delta();
	delta_original.assign(delta.begin(), delta.end());
}

// ---------------------------------------------
//Pre-compute E_original and E_new, the distance between vertex i and its neighbors
void get_E() {
	for (int i = 0; i < num_vertices; i++) {

		for (int j = 0; j < connectivity[i].size(); j++) {
			int vertex_j = connectivity[i][j];

			E_original[i][j].x = mesh_original->vertices[3 * (i + 1) + 0] - mesh_original->vertices[3 * (vertex_j + 1) + 0];
			E_original[i][j].y = mesh_original->vertices[3 * (i + 1) + 1] - mesh_original->vertices[3 * (vertex_j + 1) + 1];
			E_original[i][j].z = mesh_original->vertices[3 * (i + 1) + 2] - mesh_original->vertices[3 * (vertex_j + 1) + 2];

			E_new[i][j].x = mesh->vertices[3 * (i + 1) + 0] - mesh->vertices[3 * (vertex_j + 1) + 0];
			E_new[i][j].y = mesh->vertices[3 * (i + 1) + 1] - mesh->vertices[3 * (vertex_j + 1) + 1];
			E_new[i][j].z = mesh->vertices[3 * (i + 1) + 2] - mesh->vertices[3 * (vertex_j + 1) + 2];

		}

	}
}

//Update E_new after each iterations
//***Determine when to terminate the iterations***
void update_E() {

	float dis = 1, buff;
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < connectivity[i].size(); j++) {
			int vertex_j = connectivity[i][j];
			E_new[i][j].x = mesh->vertices[3 * (i + 1) + 0] - mesh->vertices[3 * (vertex_j + 1) + 0];
			E_new[i][j].y = mesh->vertices[3 * (i + 1) + 1] - mesh->vertices[3 * (vertex_j + 1) + 1];
			E_new[i][j].z = mesh->vertices[3 * (i + 1) + 2] - mesh->vertices[3 * (vertex_j + 1) + 2];
			buff = abs(E_new[i][j].x) + abs(E_new[i][j].y) + abs(E_new[i][j].z);
			if (buff < dis)
				dis = buff;
		}
	}

	if (abs(dis_smallest - dis) > dis_limit) {
		dis_smallest = dis;
	}
	else {
		iteration = false;
	}
}

//Return the rotation Matrix3f R
Matrix3f get_R(int vertex_i) {

	int conn_size = connectivity[vertex_i].size();

	Matrix3f Si = Matrix3f::Zero();

	vector3 ei_original, ei_new;

	for (int j = 0; j < conn_size; j++) {

		ei_original = vector3(E_original[vertex_i][j].x, E_original[vertex_i][j].y, E_original[vertex_i][j].z);
		ei_new = vector3(E_new[vertex_i][j].x, E_new[vertex_i][j].y, E_new[vertex_i][j].z);

		for (int k = 0; k < 3; k++) {
			for (int m = 0; m < 3; m++) {
				Si(k, m) += ei_original[k] * ei_new[m];
			}
		}

	}

	//SVD
	JacobiSVD<MatrixXf> svd(Si, ComputeFullU | ComputeFullV);
	const MatrixXf U = svd.matrixU();
	const MatrixXf V = svd.matrixV();	// note that this is actually V^T!!
	const MatrixXf S = svd.singularValues();

	Matrix3f Ri;

	Ri = V*(U.transpose());

	return Ri;
}

void get_triangles_position() {

	int store_idx;

	for (int i = 0; i < num_triangles; i++) {
		vector3 buffer(0, 0, 0);

		for (int j = 0; j < 3; j++) {
			store_idx = mesh->triangles[(i)].vindices[j] - 1;
			for (int k = 0; k < 3; k++) {
				buffer[k] += mesh->vertices[3 * (store_idx + 1) + k];
			}
		}

		buffer /= 3;
		for (int j = 0; j < 3; j++) {
			triangles_position[i][j] = buffer[j];
		}
	}

}

// ---------------------------------------------
//test to determine whether the cotangent delta is correct
void LeastSquareSolver_cot() {
	LeastSquaresSparseSolver solver;
	int num = num_vertices + allVertex_notInHandle.size();
	solver.Create(num, num_vertices, 3);

	float **b = new float*[3];
	b[0] = new float[num];
	b[1] = new float[num];
	b[2] = new float[num];
	//--------------------------------------------------------------
	float conn_size;
	//Laplacian -> connectivity
	for (int i = 0; i < num_vertices; i++) {
		solver.AddSysElement(i, i, 1.0f);
		//conn_size = (-1 / (float)(connectivity[i].size()));
		for (int j = 0; j < connectivity[i].size(); j++) {
			conn_size = cot_connectivity[i][j].weight;
			solver.AddSysElement(i, connectivity[i][j], conn_size);
		}
		//Delta
		for (int j = 0; j < 3; j++) {
			b[j][i] = cot_delta[i][j];
		}
	}

	int currentSize = num_vertices;
	/**/
	//Positional Constraint - original position of all vertices that are not in "handle"
	for (int i = 0; i < allVertex_notInHandle.size(); i++) {
		solver.AddSysElement(currentSize, allVertex_notInHandle[i], 1.0f);
		for (int j = 0; j < 3; j++) {
			b[j][currentSize] = mesh->vertices[3 * (allVertex_notInHandle[i] + 1) + j];
		}
		currentSize++;
	}

	solver.SetRightHandSideMatrix(b);

	// direct solver
	solver.CholoskyFactorization();
	solver.CholoskySolve(0);
	solver.CholoskySolve(1);
	solver.CholoskySolve(2);

	// get result
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < 3; j++) {
			resultList[i][j] = solver.GetSolution(j, i);
		}
	}

	// release
	solver.ResetSolver(0, 0, 0);

	delete[] b[0];
	delete[] b[1];
	delete[] b[2];
	delete[] b;
}

float getCot_Value(int pSrc, int p0, int p1) {
	float angle = 0;

	float x0 = mesh->vertices[3 * (pSrc + 1) + 0], y0 = mesh->vertices[3 * (pSrc + 1) + 1], z0 = mesh->vertices[3 * (pSrc + 1) + 2];
	float x1 = mesh->vertices[3 * (p0 + 1) + 0], y1 = mesh->vertices[3 * (p0 + 1) + 1], z1 = mesh->vertices[3 * (p0 + 1) + 2];
	float x2 = mesh->vertices[3 * (p1 + 1) + 0], y2 = mesh->vertices[3 * (p1 + 1) + 1], z2 = mesh->vertices[3 * (p1 + 1) + 2];

	float va_x = x1 - x0;
	float va_y = y1 - y0;
	float va_z = z1 - z0;
	float vb_x = x2 - x0;
	float vb_y = y2 - y0;
	float vb_z = z2 - z0;

	float va_val = sqrt(va_x*va_x + va_y*va_y + va_z*va_z);
	float vb_val = sqrt(vb_x*vb_x + vb_y*vb_y + vb_z*vb_z);

	float cosProductValue = (va_x * vb_x) + (va_y * vb_y) + (va_z * vb_z);
	float cosValue = cosProductValue / (va_val * vb_val);

	// y1z2 - z1y2, z1x2 - x1z2, x1y2 - y1x2
	vector3 sinV(va_y*vb_z - va_z*vb_y, va_z*vb_x - va_x*vb_z, va_x*vb_y - va_y*vb_x);
	float sinProductValue = sinV.length();
	float sinValue = sinProductValue / (va_val * vb_val);
	//cout << "cosValue= " << cosValue << ", sinValue" << sinValue << endl;

	float cotValue = cosValue / sinValue;
	return cotValue;
}

int getConnectivity_j(int i, int j) {
	for (int k = 0; k < connectivity[i].size(); k++) {
		if (connectivity[i][k] == j)
			return k;
	}
	return -1;
}

void getCot_Connectivity() {

	float cotValue;
	int idx_0, idx_1, idx_2;

	for (int i = 0; i < num_triangles; i++) {
		idx_0 = mesh->triangles[(i)].vindices[0] - 1;
		idx_1 = mesh->triangles[(i)].vindices[1] - 1;
		idx_2 = mesh->triangles[(i)].vindices[2] - 1;

		cotValue = getCot_Value(idx_0, idx_1, idx_2);
		cot_connectivity[idx_1][getConnectivity_j(idx_1, idx_0)].cotangent.push_back(cotValue);
		cot_connectivity[idx_2][getConnectivity_j(idx_2, idx_0)].cotangent.push_back(cotValue);

		cotValue = getCot_Value(idx_1, idx_2, idx_0);
		cot_connectivity[idx_2][getConnectivity_j(idx_2, idx_1)].cotangent.push_back(cotValue);
		cot_connectivity[idx_0][getConnectivity_j(idx_0, idx_1)].cotangent.push_back(cotValue);

		cotValue = getCot_Value(idx_2, idx_1, idx_0);
		cot_connectivity[idx_1][getConnectivity_j(idx_1, idx_2)].cotangent.push_back(cotValue);
		cot_connectivity[idx_0][getConnectivity_j(idx_0, idx_2)].cotangent.push_back(cotValue);
	}
}

void getCot_Delta() {

	float totalWeight;
	vector3 pt_0, pt_1;
	vector3 weight_cross(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < num_vertices; i++) {
		pt_0 = vector3(mesh->vertices[3 * (i + 1) + 0], mesh->vertices[3 * (i + 1) + 1], mesh->vertices[3 * (i + 1) + 2]);

		totalWeight = 0;
		weight_cross = vector3(0.0f, 0.0f, 0.0f);

		for (int j = 0; j < cot_connectivity[i].size(); j++) {
			pt_1 = vector3(mesh->vertices[3 * (connectivity[i][j] + 1) + 0], mesh->vertices[3 * (connectivity[i][j] + 1) + 1], mesh->vertices[3 * (connectivity[i][j] + 1) + 2]);

			cot_connectivity[i][j].weight = cot_connectivity[i][j].cotangent[0] + cot_connectivity[i][j].cotangent[1];
			totalWeight += cot_connectivity[i][j].weight;
			weight_cross += cot_connectivity[i][j].weight*pt_1;
		}

		for (int j = 0; j < cot_connectivity[i].size(); j++) {
			cot_connectivity[i][j].weight /= -totalWeight;
		}

		cot_delta[i] = pt_0 - (weight_cross / totalWeight);
	}

}

void getContangent() {
	getCot_Connectivity();
	getCot_Delta();
	cot_connectivity_original.assign(cot_connectivity.begin(), cot_connectivity.end());
	cot_delta_update.assign(cot_delta.begin(), cot_delta.end());
	cot_delta_original.assign(cot_delta.begin(), cot_delta.end());
}

// -------------------------------- Silhouette sketch --------------------------------
// ---------------------------------------------
// Caluculated and Restruction function
void unHandleVertex() {
	// for vertices that not in handle
	allVertex_notInHandle.clear();
	vector<int> allInHandle;
	for (int i = 0; i<handles.size(); i++)
	{
		for (int j = 0; j<handles[i].size(); j++)
		{
			if (storeOrNot(&allInHandle, handles[i][j])) {
				allInHandle.push_back(handles[i][j]);
			}
		}
	}
	sort(allInHandle.begin(), allInHandle.end());
	int j = 0;
	for (int i = 0; i < allInHandle.size(); i++) {
		while (j < num_vertices) {
			j++;
			if (allInHandle[i] == (j - 1)) {
				break;
			}
			else {
				allVertex_notInHandle.push_back((j - 1));
			}
		}
	}
	for (int i = j; i < num_vertices; i++) {
		allVertex_notInHandle.push_back(i);
	}
}

void getSolverIdx() {

	bool exist = false;
	if (!storeOrNot(&SilIdx, selected_handle_id)) {
		exist = true;
	}

	int handleItr = selected_handle_id;
	if (exist) {
		// old handle editing
		num_SilVertex += (co_silVertex.size() - SilVertex[handleItr].size());
		num_Silhouette += (co_silhouette.size() - Silhouette[handleItr].size());
	}
	else {
		// new handle editing
		num_SilVertex += co_silVertex.size();
		num_Silhouette += co_silhouette.size();
		SilIdx.push_back(handleItr);
	}

	num_SilIdx = SilIdx.size();
	num_UncurrentHandle = 0;

	for (int i = 0; i < handles.size(); i++) {
		if (i != handleItr) {
			num_UncurrentHandle += handles[i].size();
		}
	}

	SilVertex[handleItr].clear();
	SilVertex[handleItr].assign(co_silVertex.begin(), co_silVertex.end());
	Silhouette[handleItr].clear();
	Silhouette[handleItr].assign(co_silhouette.begin(), co_silhouette.end());
	/*
	cout << "SilIdx.size()= " << SilIdx.size() << endl;
	cout << "SilVertex.size()= " << SilVertex.size() << endl;
	cout << "Silhouette.size()= " << Silhouette.size() << endl;
	cout << "co_silVertex.size()= " << co_silVertex.size() << endl;
	cout << "co_silhouette.size()= "<<co_silhouette.size() << endl;
	*/
}

void solverTest(){
	Edge e;
	matrix44 m;
	vector4 vec = vector4(0.5, 0.0, 0.0, 1.0);

	gettbMatrix((float *)&m);
	vec = m * vec;
	sort(co_silVertex.begin(), co_silVertex.end());

	co_silhouette.clear();
	//co_silhouette.resize(co_silVertex.size());
	// deform handle points
	for (int i = 0; i < co_silVertex.size(); i++)
	{
		int idx = co_silVertex[i] + 1;
		e.v1 = idx - 1;
		vector3 pt(mesh->vertices[3 * idx + 0] + vec.x, mesh->vertices[3 * idx + 1] + vec.y, mesh->vertices[3 * idx + 2] + vec.z);
		e.target = pt;
		co_silhouette.push_back(e);
	}
}

void getOriginalDelta() {
	delta.clear();
	delta.assign(delta_original.begin(), delta_original.end());
	cout << "delta= " << delta.size() << endl;
	cout << "delta_original= " << delta_original.size() << endl;
}

void updateDelta() {
	float conn_size;
	for (int i = 0; i < num_vertices; i++) {

		vector3 buffer(0, 0, 0);
		conn_size = (-1 / (float)(connectivity[i].size()));

		for (int j = 0; j < connectivity[i].size(); j++) {
			for (int k = 0; k < 3; k++) {
				buffer[k] += mesh->vertices[3 * (connectivity[i][j] + 1) + k];
			}
		}

		for (int k = 0; k < 3; k++) {
			delta[i][k] = mesh->vertices[3 * (i + 1) + k] + conn_size*buffer[k];
		}
	}
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < 3; j++){
			mesh->vertices[3 * (i + 1) + j] = mesh_original->vertices[3 * (i + 1) + j];
		}
	}
}

void LeastSquareSolver_v2() {
	LeastSquaresSparseSolver solver;
	int num = num_vertices + num_UncurrentHandle + num_SilVertex + num_Silhouette;
	solver.Create(num, num_vertices, 3);

	float **b = new float*[3];
	b[0] = new float[num];
	b[1] = new float[num];
	b[2] = new float[num];
	//cout << "ALLSIZE= " << ", " << num_UncurrentHandle << ", " << handles[selected_handle_id].size() << ", " << num_SilVertex << ", " << num_Silhouette << endl;
	//--------------------------------------------------------------
	float conn_size;
	//Laplacian -> connectivity
	for (int i = 0; i < num_vertices; i++) {
		solver.AddSysElement(i, i, 1.0f);
		//conn_size = (-1 / (float)(connectivity[i].size()));
		for (int j = 0; j < connectivity[i].size(); j++) {
			conn_size = cot_connectivity[i][j].weight;
			solver.AddSysElement(i, connectivity[i][j], conn_size);
		}
		//Delta
		for (int j = 0; j < 3; j++) {
			b[j][i] = cot_delta[i][j];
		}
	}

	int currentSize = num_vertices;

	/**/
	//Positional Constraint - original position of all vertices in "handles", except current selected_handle_id
	int idx;
	for (int i = 0; i<handles.size(); i++)
	{
		if (i != selected_handle_id) {
			for (int j = 0; j<handles[i].size(); j++)
			{
				idx = handles[i][j] - 1;
				solver.AddSysElement(currentSize, idx, 0.1*Weight_unCurrentHandle.weight);
				for (int k = 0; k < 3; k++) {
					b[k][currentSize] = 0.1*Weight_unCurrentHandle.weight*mesh->vertices[3 * (idx + 1) + k];
				}
				currentSize++;
			}
		}
	}

	/**/
	//Positional Constraint - original position for vertex in "co_silhouette"
	for (int i = 0; i < SilVertex.size(); i++) {
		for (int j = 0; j < SilVertex[i].size(); j++) {
			solver.AddSysElement(currentSize, SilVertex[i][j], 0.1*Weight_targetOri.weight);
			for (int k = 0; k < 3; k++) {
				b[k][currentSize] = 0.1* Weight_targetOri.weight * mesh->vertices[3 * (SilVertex[i][j] + 1) + k];
			}
			currentSize++;
		}
	}

	/**/
	//Positional Constraint - silhouette position for co_silhouette_position, size = co_silhouette.size()
	for (int i = 0; i < Silhouette.size(); i++) {
		for (int j = 0; j < Silhouette[i].size(); j++) {
			solver.AddSysElement(currentSize, Silhouette[i][j].v1, 0.1*Weight_target.weight* Silhouette[i][j].w1);
			solver.AddSysElement(currentSize, Silhouette[i][j].v2, 0.1*Weight_target.weight* Silhouette[i][j].w2);
			for (int k = 0; k < 3; k++) {
				b[k][currentSize] = 0.1*Weight_target.weight*Silhouette[i][j].target[k];
			}
			currentSize++;
		}
	}

	solver.SetRightHandSideMatrix(b);

	// direct solver
	solver.CholoskyFactorization();
	solver.CholoskySolve(0);
	solver.CholoskySolve(1);
	solver.CholoskySolve(2);

	// get result
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < 3; j++) {
			resultList[i][j] = solver.GetSolution(j, i);
		}
	}

	// release
	solver.ResetSolver(0, 0, 0);

	delete[] b[0];
	delete[] b[1];
	delete[] b[2];
	delete[] b;
}

void LeastSquareSolver() {
	LeastSquaresSparseSolver solver;
	int num = num_vertices + num_UncurrentHandle + num_SilVertex + num_Silhouette;
	solver.Create(num, num_vertices, 3);

	float **b = new float*[3];
	b[0] = new float[num];
	b[1] = new float[num];
	b[2] = new float[num];
	//cout << "ALLSIZE= " << ", " << num_UncurrentHandle << ", " << handles[selected_handle_id].size() << ", " << num_SilVertex << ", " << num_Silhouette << endl;
	//--------------------------------------------------------------
	float conn_size;
	vector3 buff;
	Matrix3Xf eij = Matrix3Xf::Zero(3, 1);
	Matrix3Xf buff_b = Matrix3Xf::Zero(3, 1);
	//Laplacian -> connectivity
	for (int i = 0; i < num_vertices; i++) {
		//Laplacian
		solver.AddSysElement(i, i, connectivity[i].size());
		//getE
		buff_b = Matrix3Xf::Zero(3, 1);

		for (int j = 0; j < connectivity[i].size(); j++) {
			//Laplacian
			solver.AddSysElement(i, connectivity[i][j], -1.0f);
			//getE
			buff = vector3(E_original[i][j].x, E_original[i][j].y, E_original[i][j].z);
			for (int k = 0; k < 3; k++) {
				eij(k, 0) = buff[k];
			}
			buff_b += (get_R(i) + get_R(connectivity[i][j])) * eij;
		}
		buff_b = 0.5 * buff_b;

		for (int j = 0; j < 3; j++) {
			b[j][i] = buff_b(j, 0);
		}
	}

	int currentSize = num_vertices;
	/**/
	//Positional Constraint - original position of all vertices in "handles", except current selected_handle_id
	int idx;
	for (int i = 0; i<handles.size(); i++)
	{
		if (i != selected_handle_id) {
			for (int j = 0; j<handles[i].size(); j++)
			{
				idx = handles[i][j] - 1;
				solver.AddSysElement(currentSize, idx, 0.1*Weight_unCurrentHandle.weight);
				for (int k = 0; k < 3; k++) {
					b[k][currentSize] = 0.1*Weight_unCurrentHandle.weight*mesh->vertices[3 * (idx + 1) + k];
				}
				currentSize++;
			}
		}
	}

	/**/
	//Positional Constraint - original position for vertex in "co_silhouette"
	for (int i = 0; i < SilVertex.size(); i++) {
		for (int j = 0; j < SilVertex[i].size(); j++) {
			solver.AddSysElement(currentSize, SilVertex[i][j], 0.1*Weight_targetOri.weight);
			for (int k = 0; k < 3; k++) {
				b[k][currentSize] = 0.1*Weight_targetOri.weight * mesh->vertices[3 * (SilVertex[i][j] + 1) + k];
			}
			currentSize++;
		}
	}

	/**/
	//Positional Constraint - silhouette position for co_silhouette_position, size = co_silhouette.size()
	for (int i = 0; i < Silhouette.size(); i++) {
		for (int j = 0; j < Silhouette[i].size(); j++) {
			solver.AddSysElement(currentSize, Silhouette[i][j].v1, 0.1*Weight_target.weight* Silhouette[i][j].w1);
			solver.AddSysElement(currentSize, Silhouette[i][j].v2, 0.1*Weight_target.weight* Silhouette[i][j].w2);
			for (int k = 0; k < 3; k++) {
				b[k][currentSize] = 0.1*Weight_target.weight*Silhouette[i][j].target[k];
			}
			currentSize++;
		}
	}

	solver.SetRightHandSideMatrix(b);

	// direct solver
	solver.CholoskyFactorization();
	solver.CholoskySolve(0);
	solver.CholoskySolve(1);
	solver.CholoskySolve(2);

	// get result
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < 3; j++) {
			resultList[i][j] = solver.GetSolution(j, i);
		}
	}

	// release
	solver.ResetSolver(0, 0, 0);

	delete[] b[0];
	delete[] b[1];
	delete[] b[2];
	delete[] b;
}

//Reconstruction mesh
void Reconstruction() {
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < 3; j++){
			mesh->vertices[3 * (i + 1) + j] = resultList[i][j];
		}
	}
	float time = time_record.PassedTime();
	cout << "Reconstruction Time= " << time << endl;
}

void BackToOriginal() {
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < 3; j++){
			mesh->vertices[3 * (i + 1) + j] = mesh_original->vertices[3 * (i + 1) + j];
		}
	}

	delta.clear();
	delta.assign(delta_original.begin(), delta_original.end());

	cot_delta.clear();
	cot_delta.assign(cot_delta_original.begin(), cot_delta_original.end());
	cot_delta_update.clear();
	cot_delta_update.assign(cot_delta_original.begin(), cot_delta_original.end());

	cot_connectivity.clear();
	cot_connectivity.assign(cot_connectivity_original.begin(), cot_connectivity_original.end());

	for (int i = 0; i < handles.size(); i++) {
		SilVertex[i].clear();
		Silhouette[i].clear();
	}

	FeaturePath.clear();

	num_UncurrentHandle = 0;
	num_SilVertex = 0;
	num_Silhouette = 0;
	num_feature = 0;
}

// -----------------------------------------------------------------
// find the triangles that handle selected
//Test for function getHandlesTriangles()
void zeroTriangles() {
	int a, b, c;
	int handleIter = handles.size() - 1;

	for (int i = 0; i < handles_triangles[handleIter].size(); i++)
	{
		int idx = handles_triangles[handleIter][i];
		a = mesh->triangles[(idx)].vindices[0] - 1;
		b = mesh->triangles[(idx)].vindices[1] - 1;
		c = mesh->triangles[(idx)].vindices[2] - 1;

		mesh->vertices[3 * (a + 1) + 0] = 0;
		mesh->vertices[3 * (a + 1) + 1] = 0;
		mesh->vertices[3 * (a + 1) + 2] = 0;
		mesh->vertices[3 * (b + 1) + 0] = 0;
		mesh->vertices[3 * (b + 1) + 1] = 0;
		mesh->vertices[3 * (b + 1) + 2] = 0;
		mesh->vertices[3 * (c + 1) + 0] = 0;
		mesh->vertices[3 * (c + 1) + 1] = 0;
		mesh->vertices[3 * (c + 1) + 2] = 0;
	}
}

//Find the triangles that are in selected handles and store in "handles_triangles" 
void getHandlesTriangles() {

	int store_idx;
	int handleIter = handles.size() - 1;
	vector<int> this_triangle, triangle;
	vector<int> triangle_times;

	//cout << "There are " << handles[handleIter].size() << " vertices" << endl;
	for (int vertIter = 0; vertIter<handles[handleIter].size(); vertIter++)
	{
		int idx = handles[handleIter][vertIter] - 1;

		for (int triangleIter = 0; triangleIter < vertexFindex[idx].size(); triangleIter++) {

			store_idx = vertexFindex[idx][triangleIter];
			// determine if triangleIter has already in this_triangle
			bool store = true;
			for (int i = 0; i < this_triangle.size(); i++) {
				if (this_triangle[i] == store_idx) {
					store = false;
					triangle_times[i]++;
				}
			}
			if (store) {
				this_triangle.push_back(store_idx);
				triangle_times.push_back(1);
			}

		}

	}

	for (int i = 0; i < this_triangle.size(); i++) {
		if (triangle_times[i] >= 3) {
			triangle.push_back(this_triangle[i]);
		}
	}

	handles_triangles.push_back(triangle);
	//zeroTriangles();
}

// -----------------------------------------------------------------
// When left button up...
// only for contour vertex
void inSilhouetteOrNot() {
	vector<int> vert;
	for (int i = 0; i < co_silhouette.size(); i++) {
		if (storeOrNot(&vert, co_silhouette[i].v1))
			vert.push_back(co_silhouette[i].v1);
		if (storeOrNot(&vert, co_silhouette[i].v2))
			vert.push_back(co_silhouette[i].v2);
	}
	sort(vert.begin(), vert.end());

	co_silVertex.clear();
	int j = 0;
	for (int i = 0; i < vert.size(); i++) {
		while (j < num_vertices) {
			j++;
			if (vert[i] == (j - 1)) {
				co_silVertex.push_back((j - 1));
				break;
			}
		}
	}
}

// Normalize length Silhouette and Contour, then find correspndence
void getContourLength() {
	// reverse contour_edge to match the order of silhouette
	vector3 con_first((contour_edge[0].p1*contour_edge[0].w1) + (contour_edge[0].p2*contour_edge[0].w2));
	vector3 con_last((contour_edge[contour_edge.size() - 1].p1*contour_edge[contour_edge.size() - 1].w1) + (contour_edge[contour_edge.size() - 1].p2*contour_edge[contour_edge.size() - 1].w2));
	vector3 mouse_first(mouse_tracking[0]);
	vector3 mouse_last(mouse_tracking[mouse_tracking.size() - 1]);

	if (((con_first - mouse_first).lengthSqr() + (con_last - mouse_last).lengthSqr())
		> ((con_first - mouse_last).lengthSqr() + (con_last - mouse_first).lengthSqr())) {
		//cout << "reverse" << endl;
		reverse(contour_edge.begin(), contour_edge.end());
	}

	float length = 0;
	contour_normal.resize(contour_edge.size());
	for (int i = 0; i < contour_edge.size(); i++) {
		contour_normal[i] = (contour_edge[i].length_dif) / lengthOfContour;
	}
}

void getSilhouetteLength() {
	// if size of silhouette is smaller than contour
	vector3 buf;
	int des;
	int contourSize = 5*contour_edge.size();
	int mouseSize = mouse_tracking.size();
	int mul = ((contourSize - mouseSize) / (mouseSize - 1)) + 1;

	if (contourSize > mouseSize) {
		for (int i = 1; i < mouse_tracking_original.size(); i++) {
			for (int j = 0; j < mul; j++) {
				des = j + 1 + (mul + 1)*(i - 1);
				buf = (mouse_tracking_original[i - 1] * (mul - j) + mouse_tracking_original[i] * (j + 1)) / (mul + 1);
				mouse_tracking.insert(mouse_tracking.begin() + des, buf);
			}
		}
	}

	/**/
	/*
	//another way to insert, but sometimes get bad memory alloc
	vector<vector3> newPoint(mul);
	if (mul > 0) {
	for (int i = 1; i < mouse_tracking_original.size(); i++) {
	for (int j = 0; j < mul; j++) {
	buf = (mouse_tracking_original[i - 1] * (mul - j) + mouse_tracking_original[i] * (j + 1)) / (mul + 1);
	newPoint[j] = buf;
	}
	mouse_tracking.insert(mouse_tracking.begin() + 1 + (mul + 1)*(i - 1), newPoint.begin(), newPoint.end());
	}
	}
	*/

	float length = 0;
	silhouette_normal.resize(mouse_tracking.size());

	silhouette_normal[0] = 0;
	for (int i = 1; i < mouse_tracking.size(); i++) {
		length += sqrt((mouse_tracking[i - 1] - mouse_tracking[i]).lengthSqr());
		silhouette_normal[i] = length;
	}
	for (int i = 1; i < mouse_tracking.size(); i++) {
		silhouette_normal[i] /= length;
	}
}

void findCorrespondence() {
	int j = 0;
	float con_value, sil_value;
	bool find = false;
	co_silhouette.clear();
	//cout << "contour_normal.size()= " << contour_normal.size() << endl;
	//cout << "silhouette_normal.size()= " << silhouette_normal.size() << endl;
	if (contour_normal[0]==1)
		reverse(contour_normal.begin(), contour_normal.end());

	for (int i = 0; i < contour_normal.size(); i++) {
		find = false;
		con_value = contour_normal[i];
		//cout << "i= " << i << ", value= " << con_value << endl;
		while (j < silhouette_normal.size()) {
			sil_value = silhouette_normal[j];
			//cout << "j= " << j << ", value= " << sil_value << endl;
			if (sil_value == con_value) {
				find = true;
				co_silhouette.push_back(contour_edge[i]);
				co_silhouette[co_silhouette.size() - 1].target = mouse_tracking[j];
			}
			else if (sil_value > con_value) {
				find = true;
				co_silhouette.push_back(contour_edge[i]);
				co_silhouette[co_silhouette.size() - 1].target = (mouse_tracking[j] + mouse_tracking[j - 1]) / 2;
			}
			
			j++;
			if (find)
				break;
		}
	}
	//cout << "co_silhouette.size()= " << co_silhouette.size() << endl;

}

void getCorrespondence() {
	getContourLength();
	getSilhouetteLength();
	//cout << "Complete: Normalize Contour and  Silhouette length" << endl;
	findCorrespondence();
	regetSilhouetteDepth();
	//cout << "Complete: Find the correspondece between Contour and Silhouette" << endl;
}

//test for "Correspondence()"
void drawCorrespondence() {
	glLineWidth(1);
	glColor3f(1.0, 0.0, 1.0);
	for (int i = 0; i < co_silhouette.size(); i++) {
		vector3 b = (co_silhouette[i].p1*co_silhouette[i].w1) + (co_silhouette[i].p2*co_silhouette[i].w2);

		glBegin(GL_LINES);
		glVertex3f(co_silhouette[i].target[0], co_silhouette[i].target[1], co_silhouette[i].target[2]);
		glVertex3f(b[0], b[1], b[2]);
		glEnd();
	}
}

// -----------------------------------------------------------------
// Silhouette related function
float get_depth(vector2 _2Dpos)
{
	//projection_helper
	float Depth;
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	//tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	glReadPixels((int)_2Dpos.x, viewport[3] - (int)_2Dpos.y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &Depth);

	double X = _2Dpos.x;
	double Y = _2Dpos.y;
	double wpos[3] = { 0.0, 0.0, 0.0 };
	
	gluUnProject(X, ((double)viewport[3] - Y), (double)Depth, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0], &wpos[1], &wpos[2]);

	return (float)Depth;
}

// Re-get the depth of silhouette by its corresponding contour vertex
void regetSilhouetteDepth() {
	float depth;
	vector3 edge, newSil;
	vector<int> uncorrect;
	vector<float> all_depth;

	for (int i = 0; i < co_silhouette.size(); i++) {
		edge = (co_silhouette[i].p1*co_silhouette[i].w1) + (co_silhouette[i].p2*co_silhouette[i].w2);
		depth = get_depth(projection_helper(edge));
		all_depth.push_back(depth);

		newSil = Unprojection_silhouettes(projection_helper(co_silhouette[i].target), depth);
		co_silhouette[i].target = newSil;

		if (depth >= 1 || depth <= 0)
			uncorrect.push_back(i);
	}

	bool correct;
	for (int i = 0; i < uncorrect.size(); i++) {
		correct = false;
		for (int j = uncorrect[i]; j < co_silhouette.size(); j++) {
			if (storeOrNot(&uncorrect, j)) {
				correct = true;
				all_depth[uncorrect[i]] = all_depth[j];
				break;
			}
		}
		if (!correct)
		for (int j = uncorrect[i]; j >= 0; j--) {
			if (storeOrNot(&uncorrect, j)) {
				correct = true;
				all_depth[uncorrect[i]] = all_depth[j];
				break;
			}
		}
		newSil = Unprojection_silhouettes(projection_helper(co_silhouette[uncorrect[i]].target), all_depth[uncorrect[i]]);
		co_silhouette[uncorrect[i]].target = newSil;
	}
}

//get the depth of silhouette that equals the depth of contour
void getSilhouetteDepth() {
	vector3 edge;
	float count = 0, totalDepth = 0;
	bool getDepth = false;

	for (int i = 0; i < contour_edge.size(); i++) {
		edge = (contour_edge[i].p1*contour_edge[i].w1) + (contour_edge[i].p2*contour_edge[i].w2);
		depthOfSilhouette = get_depth(projection_helper(edge));
		if (depthOfSilhouette < 1 && depthOfSilhouette > 0) {
			count += 1;
			totalDepth += depthOfSilhouette;
			getDepth = true;
		}
	}
	depthOfSilhouette = totalDepth / count;

}

//test for first drawing silhouette, the depth of all vertices are the same
void DrawFirstSilhouette() {
	/*
	for (int i = 1; i<mouse_tracking_draw.size(); i++) {
	glLineWidth(5.0f);
	glBegin(GL_LINES);
	glColor3f(0.0, 1.0, 1.0);
	glVertex3f(mouse_tracking_draw[i - 1][0], mouse_tracking_draw[i - 1][1], mouse_tracking_draw[i - 1][2]);	//previous mouse position
	glVertex3f(mouse_tracking_draw[i][0], mouse_tracking_draw[i][1], mouse_tracking_draw[i][2]);	//current mouse position
	glVertex3f(0.0, 0.0, 0.0);
	glEnd();
	}
	*/
	for (int i = 1; i<mouse_tracking.size(); i++) {
		glLineWidth(2.0f);
		glBegin(GL_LINES);
		glColor3f(0.0, 1.0, 1.0);
		glVertex3f(mouse_tracking[i - 1][0], mouse_tracking[i - 1][1], mouse_tracking[i - 1][2]);	//previous mouse position 
		glVertex3f(mouse_tracking[i][0], mouse_tracking[i][1], mouse_tracking[i][2]);	//current mouse position
		glVertex3f(0.0, 0.0, 0.0);
		glEnd();
	}
}

void DrawSilhouette() {
	for (int i = 1; i<co_silhouette.size(); i++) {
		glLineWidth(2.0f);
		glBegin(GL_LINES);
		glColor3f(0.0, 1.0, 0.0);
		glVertex3f(co_silhouette[i - 1].target[0], co_silhouette[i - 1].target[1], co_silhouette[i - 1].target[2]);	//previous mouse position 
		glVertex3f(co_silhouette[i].target[0], co_silhouette[i].target[1], co_silhouette[i].target[2]);	//current mouse position
		glVertex3f(0.0, 0.0, 0.0);
		glEnd();
	}
}

void DrawSilhouette_test() {
	for (int i = 0; i<co_silhouette.size(); i++) {
		glLineWidth(5.0f);
		glBegin(GL_LINES);
		glColor3f(0.0, 1.0, 0.0);
		glVertex3f(co_silhouette[i].target[0], co_silhouette[i].target[1], co_silhouette[i].target[2]);	//current mouse position
		glVertex3f(co_silhouette[i].target[0] + 0.01, co_silhouette[i].target[1], co_silhouette[i].target[2]);	//current mouse position
		glVertex3f(0.0, 0.0, 0.0);
		glEnd();
	}
}
// -----------------------------------------------------------------
// Contour Fuctions
vector3 getPosition(int idx) {
	vector3 pt(mesh->vertices[3 * (idx + 1) + 0], mesh->vertices[3 * (idx + 1) + 1], mesh->vertices[3 * (idx + 1) + 2]);
	return pt;
}

Edge getEdge(int v1, int v2) {
	Edge edge;
	edge.v1 = v1;
	edge.v2 = v2;
	edge.p1 = getPosition(v1);
	edge.p2 = getPosition(v2);
	return edge;
}

//find triangles contain v1 and v2, but is not idx_tri
int findAotherTriangle(int idx_tri, int v1, int v2) {
	int a, b, c;
	vector<int> in_triangle;

	// find which triangle contains v1 except idx_tri
	for (int i = 0; i < handles_contour_triangles.size(); i++) {
		int idx_tri_2 = handles_contour_triangles[i];
		a = mesh->triangles[(idx_tri_2)].vindices[0] - 1;
		b = mesh->triangles[(idx_tri_2)].vindices[1] - 1;
		c = mesh->triangles[(idx_tri_2)].vindices[2] - 1;

		if (idx_tri != idx_tri_2) {
			if (a == v1 || b == v1 || c == v1) {
				if (storeOrNot(&in_triangle, idx_tri_2))
					in_triangle.push_back(idx_tri_2);
			}
		}
	}

	for (int i = 0; i < in_triangle.size(); i++) {
		int idx_tri_2 = in_triangle[i];
		for (int j = 0; j < vertexFindex[v2].size(); j++) {
			if (vertexFindex[v2][j] == in_triangle[i]) {
				if (storeOrNot(&contour_triangles, idx_tri_2)) {
					contour_triangles.push_back(idx_tri_2);
					return idx_tri_2;
				}
				else
					return -1;
			}

		}
	}

	return -1;
}

//find edge connect positive vertex and negative vertex, and haven't been through  
void findEdge(int mode, int idx_tri, int v1, int v2) {

	if (idx_tri != -1) {
		Edge e;
		int idx_tri_2;
		int a, b, c;

		a = mesh->triangles[(idx_tri)].vindices[0] - 1;
		b = mesh->triangles[(idx_tri)].vindices[1] - 1;
		c = mesh->triangles[(idx_tri)].vindices[2] - 1;

		if (vertex_value[a] * vertex_value[b] < 0 && vertex_value[a] * vertex_value[c] < 0) { // a is different
			if (b == v1 || b == v2) {
				e = getEdge(a, b);
				idx_tri_2 = findAotherTriangle(idx_tri, a, c);
				findEdge(mode, idx_tri_2, a, c);
			}
			else {
				e = getEdge(a, c);
				idx_tri_2 = findAotherTriangle(idx_tri, a, b);
				findEdge(mode, idx_tri_2, a, b);
			}
		}
		else if (vertex_value[b] * vertex_value[a] < 0 && vertex_value[b] * vertex_value[c] < 0) { // b is different
			if (a == v1 || a == v2) {
				e = getEdge(b, a);
				idx_tri_2 = findAotherTriangle(idx_tri, b, c);
				findEdge(mode, idx_tri_2, b, c);
			}
			else {
				e = getEdge(b, c);
				idx_tri_2 = findAotherTriangle(idx_tri, b, a);
				findEdge(mode, idx_tri_2, b, a);
			}
		}
		else if (vertex_value[c] * vertex_value[a] < 0 && vertex_value[c] * vertex_value[b] < 0) { // c is different
			if (a == v1 || a == v2) {
				e = getEdge(c, a);
				idx_tri_2 = findAotherTriangle(idx_tri, c, b);
				findEdge(mode, idx_tri_2, c, b);
			}
			else {
				e = getEdge(c, b);
				idx_tri_2 = findAotherTriangle(idx_tri, c, a);
				findEdge(mode, idx_tri_2, c, a);
			}
		}
		if (mode == 0)
			first_contour_edge.push_back(e);
		else
			reverse_contour_edge.push_back(e);
	}
}

//Start from "startPoint", find contour in order 
void first_findEdge(int startPoint) {
	//problem ~ force it find only one contour
	Edge e;
	int idx_tri = handles_contour_triangles[startPoint];
	int idx_tri_2;
	int a, b, c;

	contour_triangles.clear();
	contour_triangles.push_back(idx_tri);

	a = mesh->triangles[(idx_tri)].vindices[0] - 1;
	b = mesh->triangles[(idx_tri)].vindices[1] - 1;
	c = mesh->triangles[(idx_tri)].vindices[2] - 1;


	if (vertex_value[a] * vertex_value[b] < 0 && vertex_value[a] * vertex_value[c] < 0) { // a is different
		e = getEdge(a, b);
		idx_tri_2 = findAotherTriangle(idx_tri, a, c);
		findEdge(0, idx_tri_2, a, c);
	}
	else if (vertex_value[b] * vertex_value[a] < 0 && vertex_value[b] * vertex_value[c] < 0) { // b is different
		e = getEdge(b, a);
		idx_tri_2 = findAotherTriangle(idx_tri, b, c);
		findEdge(0, idx_tri_2, b, c);
	}
	else if (vertex_value[c] * vertex_value[a] < 0 && vertex_value[c] * vertex_value[b] < 0) { // c is different
		e = getEdge(c, a);
		idx_tri_2 = findAotherTriangle(idx_tri, c, b);
		findEdge(0, idx_tri_2, c, b);
	}

	first_contour_edge.push_back(e);
}

//Start from "startPoint", find contour reversely
void reverse_findEdge(int startPoint) {
	//problem ~ force it find only one contour
	Edge e;
	int idx_tri = handles_contour_triangles[startPoint];
	int idx_tri_2;
	int a, b, c;


	a = mesh->triangles[(idx_tri)].vindices[0] - 1;
	b = mesh->triangles[(idx_tri)].vindices[1] - 1;
	c = mesh->triangles[(idx_tri)].vindices[2] - 1;

	if (vertex_value[a] * vertex_value[b] < 0 && vertex_value[a] * vertex_value[c] < 0) { // a is different
		e = getEdge(a, c);
		idx_tri_2 = findAotherTriangle(idx_tri, a, b);
		findEdge(1, idx_tri_2, a, b);
	}
	else if (vertex_value[b] * vertex_value[a] < 0 && vertex_value[b] * vertex_value[c] < 0) { // b is different
		e = getEdge(b, c);
		idx_tri_2 = findAotherTriangle(idx_tri, b, a);
		findEdge(1, idx_tri_2, b, a);
	}
	else if (vertex_value[c] * vertex_value[a] < 0 && vertex_value[c] * vertex_value[b] < 0) { // c is different
		e = getEdge(c, b);
		idx_tri_2 = findAotherTriangle(idx_tri, c, a);
		findEdge(1, idx_tri_2, c, a);
	}
	reverse_contour_edge.push_back(e);
}

//Find contour triangles of handle that currently selected, and return its size
int getContourTriangles() {

	handles_vertex.clear();
	contour_edge.clear();
	handles_contour_triangles.clear();

	vector<int> handles_vertex_value;
	vector<int> positive_triangle, negative_triangle;

	int a, b, c;
	int handleIter = selected_handle_id;
	if (handleIter < 0)
		handleIter = handles_triangles.size() - 1;

	for (int i = 0; i < handles_triangles[handleIter].size(); i++)
	{
		int idx_tri = handles_triangles[handleIter][i];

		//store vertices that in this triangle in "all_vertex" and set its inital value = 0
		for (int j = 0; j < 3; j++) {
			int idx = mesh->triangles[(idx_tri)].vindices[j] - 1;
			if (storeOrNot(&handles_vertex, idx)) {
				handles_vertex.push_back(idx);
				handles_vertex_value.push_back(0);
			}
		}

		float buffer = 0;
		buffer += (eyex - triangles_position[idx_tri][0]) * mesh->facetnorms[3 * (idx_tri + 1) + 0];
		buffer += (eyey - triangles_position[idx_tri][1]) * mesh->facetnorms[3 * (idx_tri + 1) + 1];
		buffer += (eyez - triangles_position[idx_tri][2]) * mesh->facetnorms[3 * (idx_tri + 1) + 2];

		//store positive triangles in "positive_triangle" and value of vertices in this triangle + 1
		if (buffer >= 0) {
			if (storeOrNot(&positive_triangle, idx_tri)) {
				positive_triangle.push_back(idx_tri);
			}
			for (int j = 0; j < 3; j++) {
				int idx = mesh->triangles[(idx_tri)].vindices[j] - 1;
				for (int k = 0; k < handles_vertex.size(); k++) {
					if (idx == handles_vertex[k]){
						handles_vertex_value[k]++;
						break;
					}
				}
			}
		}
		//store negative triangles in "negative_triangle" and value of vertices in this triangle - 1
		else if (buffer <= 0) {
			if (storeOrNot(&negative_triangle, idx_tri)) {
				negative_triangle.push_back(idx_tri);
			}
			for (int j = 0; j < 3; j++) {
				int idx = mesh->triangles[(idx_tri)].vindices[j] - 1;
				for (int k = 0; k < handles_vertex.size(); k++) {
					if (idx == handles_vertex[k]){
						handles_vertex_value[k]--;
						break;
					}
				}
			}
		}
	}

	// store handle_vertex_value into vertex_value, to find the value more easier
	vertex_value_ori.resize(num_vertices, -100);
	vertex_value.resize(num_vertices, -100);
	for (int i = 0; i < handles_vertex.size(); i++) {
		vertex_value_ori[handles_vertex[i]] = handles_vertex_value[i];
		vertex_value[handles_vertex[i]] = handles_vertex_value[i];
		if (handles_vertex_value[i] == 0) {
			vertex_value[handles_vertex[i]]--;
		}
	}

	// getHandlesContourTriangles, store only the triangle that will contains contour
	vector3 triangle(0, 0, 0);
	for (int i = 0; i < handles_triangles[handleIter].size(); i++) {
		int idx_tri = handles_triangles[handleIter][i];

		a = mesh->triangles[(idx_tri)].vindices[0] - 1;
		b = mesh->triangles[(idx_tri)].vindices[1] - 1;
		c = mesh->triangles[(idx_tri)].vindices[2] - 1;

		if (vertex_value[a]>0 && vertex_value[b] > 0 && vertex_value[c] > 0) {
			//none
		}
		else if (vertex_value[a] < 0 && vertex_value[b] < 0 && vertex_value[c] < 0) {
			//none
		}
		else {
			handles_contour_triangles.push_back(idx_tri);
		}
	}
	return handles_contour_triangles.size();
}

//Combine "first_contour_edge" and "reverse_contour_edge"
void combine_Edge() {
	//combime "first_contour_edge" and "reverse_contour_edge", assign to "contour_edge"
	if (first_contour_edge.size() > reverse_contour_edge.size()) {
		contour_edge.assign(first_contour_edge.begin(), first_contour_edge.end());
		for (int i = reverse_contour_edge.size() - 1; i >= 0; i--) {
			contour_edge.push_back(reverse_contour_edge[i]);
		}
	}
	else {
		contour_edge.assign(reverse_contour_edge.begin(), reverse_contour_edge.end());
		for (int i = first_contour_edge.size() - 1; i >= 0; i--) {
			contour_edge.push_back(first_contour_edge[i]);
		}
	}
	first_contour_edge.clear();
	reverse_contour_edge.clear();
}

void getWeightOfEdge() {
	float abs_1, abs_2;
	for (int i = 0; i < contour_edge.size(); i++) {
		if (vertex_value_ori[contour_edge[i].v1] == 0) {
			// value of vertex v1 = 0
			contour_edge[i].w1 = 1.0f;
			contour_edge[i].w2 = 0.0f;
		}
		else if (vertex_value_ori[contour_edge[i].v2] == 0) {
			// value of vertex v2 = 0
			contour_edge[i].w1 = 0.0f;
			contour_edge[i].w2 = 1.0f;
		}
		else {
			// both value of two vertices not equal 0
			abs_1 = abs(vertex_value_ori[contour_edge[i].v1]);
			abs_2 = abs(vertex_value_ori[contour_edge[i].v2]);
			contour_edge[i].w1 = abs_2 / (abs_1 + abs_2);
			contour_edge[i].w2 = abs_1 / (abs_1 + abs_2);
		}
	}
}

void getLengthOfEdge() {
	lengthOfContour = 0;
	float length = 0;
	vector3 a, b;
	contour_edge[0].length_dif = 0;

	for (int i = 1; i < contour_edge.size(); i++) {
		a = (contour_edge[i].p1*contour_edge[i].w1) + (contour_edge[i].p2*contour_edge[i].w2);
		b = (contour_edge[i - 1].p1*contour_edge[i - 1].w1) + (contour_edge[i - 1].p2*contour_edge[i - 1].w2);
		length += sqrt((a - b).lengthSqr());
		lengthOfContour += length;
		contour_edge[i].length_dif = lengthOfContour;
	}
}

void findContour() {
	if (handles.size() > 0) {
		contourTriangle_size = getContourTriangles();
		firstFindContour(0);
	}
}

void firstFindContour(int startPoint) {
	refind = vector2(selected_handle_id, startPoint);
	first_findEdge(startPoint);
	reverse_findEdge(startPoint);
	combine_Edge();
	getWeightOfEdge();
	getLengthOfEdge();
	//get the depth of silhouette
	getSilhouetteDepth();

	/*
	cout << "contour_edge= " << endl;
	for (int i = 1; i < contour_edge.size(); i++) {
		cout << "i= " << i << ", " << contour_edge[i].v1 << " " << contour_edge[i].v2 << endl;
	}
	*/
}

void reFindContour(int startPoint) {
	first_findEdge(startPoint);
	reverse_findEdge(startPoint);
	combine_Edge();
	getWeightOfEdge();
	getLengthOfEdge();
	//get the depth of silhouette
	getSilhouetteDepth();
}

void drawContour() {
	// test
	vector3 a, b;
	glLineWidth(8);
	glColor3f(1.0, 0.0, 0.0);
	for (int i = 1; i < contour_edge.size(); i++) {
		if (i == 1 || i == (contour_edge.size() - 1))
			glColor3f(1.0, 1.0, 0.0);
		else
			glColor3f(1.0, 0.0, 0.0);
		
		int c_v1 = contour_edge[i].v1;
		int c_v2 = contour_edge[i].v2;
		int p_v1 = contour_edge[i-1].v1;
		int p_v2 = contour_edge[i-1].v2;
		vector3 cp1(mesh->vertices[3 * (c_v1 + 1) + 0], mesh->vertices[3 * (c_v1 + 1) + 1], mesh->vertices[3 * (c_v1 + 1) + 2]);
		vector3 cp2(mesh->vertices[3 * (c_v2 + 1) + 0], mesh->vertices[3 * (c_v2 + 1) + 1], mesh->vertices[3 * (c_v2 + 1) + 2]);
		vector3 pp1(mesh->vertices[3 * (p_v1 + 1) + 0], mesh->vertices[3 * (p_v1 + 1) + 1], mesh->vertices[3 * (p_v1 + 1) + 2]);
		vector3 pp2(mesh->vertices[3 * (p_v2 + 1) + 0], mesh->vertices[3 * (p_v2 + 1) + 1], mesh->vertices[3 * (p_v2 + 1) + 2]);

		a = (cp1*contour_edge[i].w1) + (cp2*contour_edge[i].w2);
		b = (pp1*contour_edge[i - 1].w1) + (pp2*contour_edge[i - 1].w2);

		glBegin(GL_LINES);
		glVertex3f(a[0], a[1], a[2]);
		glVertex3f(b[0], b[1], b[2]);
		glEnd();
	}
}

// --------------------------------- Feature sketch ----------------------------------
// ---------------------------------------------
int getPathIdx(int idx) {
	for (int i = 0; i < feature_path_vertices.size(); i++) {
		if (feature_path_vertices[i] == idx){
			return i;
		}
	}
	return -1;
}

void LeastSquareSolver_update() {
	LeastSquaresSparseSolver solver;
	//int num = num_vertices + num_cp;
	int num = num_vertices + num_cp;
	solver.Create(num, num_vertices, 3);

	float **b = new float*[3];
	b[0] = new float[num];
	b[1] = new float[num];
	b[2] = new float[num];

	float conn_size;
	//Laplacian -> connectivity
	for (int i = 0; i < num_vertices; i++) {
		solver.AddSysElement(i, i, 1.0f);
		//conn_size = (-1 / (float)(connectivity[i].size()));
		for (int j = 0; j < connectivity[i].size(); j++) {
			conn_size = cot_connectivity[i][j].weight;
			solver.AddSysElement(i, connectivity[i][j], conn_size);
		}
		//Delta
		for (int j = 0; j < 3; j++) {
			b[j][i] = cot_delta[i][j];
		}
	}


	int currentSize = num_vertices;
	for (int i = 0; i < num_cp; i++) {
		solver.AddSysElement(currentSize, randomCpList[i], 1.0f);
		for (int j = 0; j < 3; j++) {
			b[j][currentSize] = mesh->vertices[3 * (randomCpList[i] + 1) + j];
		}
		currentSize++;
	}

	solver.SetRightHandSideMatrix(b);
	// direct solver
	solver.CholoskyFactorization();
	solver.CholoskySolve(0);
	solver.CholoskySolve(1);
	solver.CholoskySolve(2);

	// get result
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < 3; j++) {
			mesh->vertices[3 * (i + 1) + j] = solver.GetSolution(j, i);
		}
	}

	solver.ResetSolver(0, 0, 0);

	delete[] b[0];
	delete[] b[1];
	delete[] b[2];
	delete[] b;
}

void scaleCotDelta() {
	for (int i = 0; i < feature_path.size(); i++) {
		for (int j = 0; j < 3; j++) {
			cot_delta[feature_path[i]][j] = (0.1 * Weight_feature.weight*cot_delta_update[feature_path[i]][j]);
		}
	}
}

void updateCotangent(vector<int> *path) {

	float cotValue;
	int tri_idx;
	int idx_0, idx_1, idx_2;
	vector<int> in_triangle;

	//clear
	for (int i = 0; i < path->size(); i++) {
		for (int j = 0; j < vertexFindex[path->at(i)].size(); j++) {
			tri_idx = vertexFindex[path->at(i)][j];

			idx_0 = mesh->triangles[(tri_idx)].vindices[0] - 1;
			idx_1 = mesh->triangles[(tri_idx)].vindices[1] - 1;
			idx_2 = mesh->triangles[(tri_idx)].vindices[2] - 1;

			cot_connectivity[idx_1][getConnectivity_j(idx_1, idx_0)].cotangent.clear();
			cot_connectivity[idx_2][getConnectivity_j(idx_2, idx_0)].cotangent.clear();

			cot_connectivity[idx_2][getConnectivity_j(idx_2, idx_1)].cotangent.clear();
			cot_connectivity[idx_0][getConnectivity_j(idx_0, idx_1)].cotangent.clear();

			cot_connectivity[idx_1][getConnectivity_j(idx_1, idx_2)].cotangent.clear();
			cot_connectivity[idx_0][getConnectivity_j(idx_0, idx_2)].cotangent.clear();
		}
	}

	// find triangle
	for (int i = 0; i < path->size(); i++) {
		for (int j = 0; j < vertexFindex[path->at(i)].size(); j++) {

			tri_idx = vertexFindex[path->at(i)][j];

			idx_0 = mesh->triangles[(tri_idx)].vindices[0] - 1;
			idx_1 = mesh->triangles[(tri_idx)].vindices[1] - 1;
			idx_2 = mesh->triangles[(tri_idx)].vindices[2] - 1;

			cotValue = getCot_Value(idx_0, idx_1, idx_2);
			cot_connectivity[idx_1][getConnectivity_j(idx_1, idx_0)].cotangent.push_back(cotValue);
			cot_connectivity[idx_2][getConnectivity_j(idx_2, idx_0)].cotangent.push_back(cotValue);

			cotValue = getCot_Value(idx_1, idx_2, idx_0);
			cot_connectivity[idx_2][getConnectivity_j(idx_2, idx_1)].cotangent.push_back(cotValue);
			cot_connectivity[idx_0][getConnectivity_j(idx_0, idx_1)].cotangent.push_back(cotValue);

			cotValue = getCot_Value(idx_2, idx_1, idx_0);
			cot_connectivity[idx_1][getConnectivity_j(idx_1, idx_2)].cotangent.push_back(cotValue);
			cot_connectivity[idx_0][getConnectivity_j(idx_0, idx_2)].cotangent.push_back(cotValue);
		}
	}


	int idx;
	float totalWeight;
	vector3 pt_0, pt_1;
	vector3 weight_cross(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < path->size(); i++) {
		idx = path->at(i);
		pt_0 = vector3(mesh->vertices[3 * (idx + 1) + 0], mesh->vertices[3 * (idx + 1) + 1], mesh->vertices[3 * (idx + 1) + 2]);

		totalWeight = 0;
		weight_cross = vector3(0.0f, 0.0f, 0.0f);

		for (int j = 0; j < cot_connectivity[idx].size(); j++) {
			pt_1 = vector3(mesh->vertices[3 * (connectivity[idx][j] + 1) + 0], mesh->vertices[3 * (connectivity[idx][j] + 1) + 1], mesh->vertices[3 * (connectivity[idx][j] + 1) + 2]);

			cot_connectivity[idx][j].weight = cot_connectivity[idx][j].cotangent[0] + cot_connectivity[idx][j].cotangent[1];
			totalWeight += cot_connectivity[idx][j].weight;
			weight_cross += cot_connectivity[idx][j].weight*pt_1;
		}

		for (int j = 0; j < cot_connectivity[idx].size(); j++) {
			cot_connectivity[idx][j].weight /= -totalWeight;
		}

		cot_delta[idx] = pt_0 - (weight_cross / totalWeight);
		cot_delta_update[idx] = pt_0 - (weight_cross / totalWeight);
	}
}

void LeastSquareSolver_feature() {
	LeastSquaresSparseSolver solver;
	int num = num_vertices + feature_path.size() + feature_path.size() - 2 + num_cp;
	solver.Create(num, num_vertices, 3);

	float **b = new float*[3];
	b[0] = new float[num];
	b[1] = new float[num];
	b[2] = new float[num];

	int idx;
	float conn_size;
	//Laplacian -> connectivity
	for (int i = 0; i < num_vertices; i++) {

		solver.AddSysElement(i, i, 1.0f);
		if (storeOrNot(&feature_roundVertices, i)) {
			for (int j = 0; j < connectivity[i].size(); j++) {
				conn_size = cot_connectivity[i][j].weight;
				solver.AddSysElement(i, connectivity[i][j], conn_size);
			}
		}
		else {
			conn_size = (-1 / (float)(connectivity[i].size()));
			for (int j = 0; j < connectivity[i].size(); j++) {
				solver.AddSysElement(i, connectivity[i][j], conn_size);
			}
		}
		//Delta
		for (int j = 0; j < 3; j++) {
			b[j][i] = cot_delta[i][j];
		}
	}

	float f_weight = 1;
	int currentSize = num_vertices;
	//Positional Constraint
	for (int i = 0; i < feature_path.size(); i++) {
		solver.AddSysElement(currentSize, feature_path[i], f_weight);
		for (int j = 0; j < 3; j++) {
			b[j][currentSize] = f_weight * mesh->vertices[3 * (feature_path[i] + 1) + j];
		}
		currentSize++;
	}

	f_weight = 3;
	//Positional Constraint - averaging constraint loosens
	for (int i = 1; i < feature_path.size() - 1; i++) {
		solver.AddSysElement(currentSize, feature_path[i - 1], -0.5*f_weight);
		solver.AddSysElement(currentSize, feature_path[i], 1.0*f_weight);
		solver.AddSysElement(currentSize, feature_path[i + 1], -0.5*f_weight);
		for (int j = 0; j < 3; j++) {
			b[j][currentSize] = 0;
		}
		currentSize++;
	}

	for (int i = 0; i < num_cp; i++) {
		solver.AddSysElement(currentSize, randomCpList[i], 1.0f);
		for (int j = 0; j < 3; j++) {
			b[j][currentSize] = mesh->vertices[3 * (randomCpList[i] + 1) + j];
		}
		currentSize++;
	}

	solver.SetRightHandSideMatrix(b);
	// direct solver
	solver.CholoskyFactorization();
	solver.CholoskySolve(0);
	solver.CholoskySolve(1);
	solver.CholoskySolve(2);

	// get result
	for (int i = 0; i < num_vertices; i++) {
		for (int j = 0; j < 3; j++) {
			mesh->vertices[3 * (i + 1) + j] = solver.GetSolution(j, i);
		}
	}

	solver.ResetSolver(0, 0, 0);

	delete[] b[0];
	delete[] b[1];
	delete[] b[2];
	delete[] b;
}

void setFeaturePosition() {
	for (int i = 0; i < feature_path.size(); i++) {
		int idx = feature_path[i];
		mesh->vertices[3 * (idx + 1) + 0] = correspond_sketch[i][0];
		mesh->vertices[3 * (idx + 1) + 1] = correspond_sketch[i][1];
		mesh->vertices[3 * (idx + 1) + 2] = correspond_sketch[i][2];
	}
}

//Select (num_cp) control point randomly, 0 <= randomNum < num_vertices
void get_RandomCp(){

	randomCpList.clear();

	srand(time(NULL));
	int count = 0, randomNum;

	while (count < num_cp) {
		randomNum = rand() % num_vertices;
		if (storeOrNot(&feature_roundVertices, randomNum) && storeOrNot(&randomCpList, randomNum)) {
			randomCpList.push_back(randomNum);
			count++;
		}
	}
}

void getVerticesInPathTriangles() {

	feature_path_vertices.clear();
	feature_path_vertices.assign(feature_path.begin(), feature_path.end());

	for (int i = 0; i < feature_path.size(); i++) {
		for (int j = 0; j < connectivity[feature_path[i]].size(); j++) {
			if (storeOrNot(&feature_path_vertices, connectivity[feature_path[i]][j])) {
				feature_path_vertices.push_back(connectivity[feature_path[i]][j]);
			}
		}
	}
	// -----------------------------------------------------------------
	feature_roundVertices.clear();
	feature_roundVertices.assign(feature_path_vertices.begin(), feature_path_vertices.end());

	for (int i = feature_path.size(); i < feature_path_vertices.size(); i++) {
		for (int j = 0; j < connectivity[feature_path_vertices[i]].size(); j++) {
			if (storeOrNot(&feature_roundVertices, connectivity[feature_path_vertices[i]][j])) {
				feature_roundVertices.push_back(connectivity[feature_path_vertices[i]][j]);
			}
		}
	}
	/*
	cout << "feature_path= " << feature_path.size() << endl;
	cout << "feature_path_vertices= " << feature_path_vertices.size() << endl;
	cout << "feature_roundVertices= " << feature_roundVertices.size() << endl;
	*/
}

// -----------------------------------------------------------------
void testFeatureSize() {
	cout << "feature_tracking.size()= " << feature_tracking.size() << endl;
	cout << "FeatureTrack.size()= " << FeatureTrack.size() << endl;
	cout << "trackingPath= " << endl;
	for (int i = 0; i < FeatureTrack.size(); i++)
		cout << FeatureTrack[i].trackingPath.size() << " ";
	cout << endl << endl;

	cout << "meshPath= " << endl;
	for (int i = 0; i < FeatureTrack.size(); i++)
		cout << FeatureTrack[i].meshPath.size() << " ";
	cout << endl;
	
	cout << "feature_path.size()= " << feature_path.size() << endl;
	cout << "correspond_sketch.size()= " << correspond_sketch.size() << endl << endl;
}

vector3 getTrackingPosition(int i, int j) {
	return feature_tracking[FeatureTrack[i].trackingPath[j]];
}

void SketchCorrespondence() {

	int meshPathSize, trackingPathSize;

	correspond_sketch.clear();
	correspond_sketch.push_back(feature_tracking[FeatureTrack[0].trackingIdx]);

	for (int i = 1; i < FeatureTrack.size(); i++) {

		vector3 track_posiion_pre = feature_tracking[FeatureTrack[i-1].trackingIdx];
		vector3 track_posiion = feature_tracking[FeatureTrack[i].trackingIdx];

		meshPathSize = FeatureTrack[i].meshPath.size();
		trackingPathSize = FeatureTrack[i].trackingPath.size();

		vector3 buf;
		if (trackingPathSize == 0) {
			for (int j = 0; j < meshPathSize; j++) {
				buf = (track_posiion_pre * (meshPathSize - j) + track_posiion * (j + 1)) / (meshPathSize + 1);
				correspond_sketch.push_back(buf);
			}
		}
		else if (meshPathSize == 0) {
			//none
		}
		else if (meshPathSize == trackingPathSize) {
			for (int j = 0; j < meshPathSize; j++) {
				correspond_sketch.push_back(getTrackingPosition(i, j));
			}
		}
		else if (meshPathSize < trackingPathSize) {
			//can optimize
			int mul = floor(trackingPathSize / meshPathSize);
			for (int j = 0; j < meshPathSize; j++) {
				correspond_sketch.push_back(getTrackingPosition(i, ((j + 1)*mul - 1)));
			}
		}
		else {
			int mul = floor((meshPathSize - trackingPathSize) / trackingPathSize);
			int rest = meshPathSize - mul;
			vector3 track_buf_pre, track_buf;

			for (int j = 0; j < trackingPathSize; j++) {
				track_buf = getTrackingPosition(i, j);
				for (int k = 0; k < mul; k++) {
					if (j == 0) {
						buf = (track_posiion_pre * (mul - j) + track_buf * (j + 1)) / (mul + 1);
					}
					else if (j == FeatureTrack[i].meshPath.size() - 1) {
						buf = (track_buf * (mul - j) + track_posiion * (j + 1)) / (mul + 1);
					}
					else {
						track_buf_pre = getTrackingPosition(i, j - 1);
						buf = (track_buf_pre * (mul - j) + track_buf * (j + 1)) / (mul + 1);
					}
					correspond_sketch.push_back(buf);
				}
				correspond_sketch.push_back(track_buf);
			}
			for (int j = 0; j < rest; j++) {
				buf = (track_buf * (rest - j) + track_posiion * (j + 1)) / (rest + 1);
				correspond_sketch.push_back(buf);
			}

		}
		correspond_sketch.push_back(track_posiion);
	}

}

int walkEdge(int a, int b) {
	for (int i = 1; i < feature_path.size(); i++) {
		if ((feature_path[i - 1] == a && feature_path[i] == b) || (feature_path[i - 1] == b && feature_path[i] == a))
			return i;
	}
	return -1;
}

int findPath(int index, int idx_0, int idx_1) {
	vector3 pt_0(mesh->vertices[3 * (idx_0 + 1) + 0], mesh->vertices[3 * (idx_0 + 1) + 1], mesh->vertices[3 * (idx_0 + 1) + 2]);
	vector3 pt_1(mesh->vertices[3 * (idx_1 + 1) + 0], mesh->vertices[3 * (idx_1 + 1) + 1], mesh->vertices[3 * (idx_1 + 1) + 2]);

	int loop_limit = 10;
	int idx_2, next_idx = idx_0, smallest_idx;
	float smallest_length = 1, length;
	vector<int> walk_path;

	while (next_idx != idx_1 && loop_limit > 0) {
		loop_limit--;
		if (loop_limit == 0)
			cout << "GOTYOI!!!!" << endl;
		smallest_length = 1;
		for (int i = 0; i < connectivity[next_idx].size(); i++) {
			idx_2 = connectivity[next_idx][i];
			vector3 pt_2(mesh->vertices[3 * (idx_2 + 1) + 0], mesh->vertices[3 * (idx_2 + 1) + 1], mesh->vertices[3 * (idx_2 + 1) + 2]);
			length = (pt_1 - pt_2).lengthSqr();

			//&& storeOrNot(&feature_path, idx_2)
			if (length < smallest_length) {
				smallest_length = length;
				smallest_idx = idx_2;
			}
		}
		walk_path.push_back(smallest_idx);
		next_idx = smallest_idx;
	}

	if (loop_limit == 0)
		return -1;

	// avoid walking the same edge
	int pop;
	bool store = true;
	for (int i = 0; i < walk_path.size(); i++) {
		pop = walkEdge(feature_path[feature_path.size() - 1], walk_path[i]);

		if (pop > -1) {
			while (feature_path.size() != pop) {
				feature_path.pop_back();
				FeatureTrack[index].meshPath.pop_back();
			}
		}
		else {
			feature_path.push_back(walk_path[i]);
			if (i != (walk_path.size() - 1))
				FeatureTrack[index].meshPath.push_back(walk_path[i]);
		}
	}
	return 0;
}

void featureTracking() {

	feature_path.clear();
	feature_path.push_back(FeatureTrack[0].meshIdx);

	int ending;
	for (int i = 1; i < FeatureTrack.size(); i++) {
		ending = findPath(i, FeatureTrack[i - 1].meshIdx, FeatureTrack[i].meshIdx);
		if (ending == -1) {
			while (FeatureTrack.size()>i) {
				FeatureTrack.pop_back();

			}
			break;
		}
	}

}

// Find nearest vertex of position in "feature_tracking", and store in "featureIdx"
void findAllNearestVertex() {
	nearestIdx.clear();
	FeatureTrack.clear();
	Feature feature;

	int smallest_idx;
	float smallest_length = 1, length;
	vector<int> tracking_path;
	
	for (int i = 0; i < feature_tracking.size(); i++) {
		vector3 feature_v = feature_tracking[i];
		smallest_length = 1;

		for (int j = 0; j < num_vertices; j++) {
			vector3 pt(mesh->vertices[3 * (j + 1) + 0], mesh->vertices[3 * (j + 1) + 1], mesh->vertices[3 * (j + 1) + 2]);
			length = (feature_v - pt).lengthSqr();

			if (length < smallest_length) {
				smallest_length = length;
				smallest_idx = j;
			}
		}
		if (storeOrNot(&nearestIdx, smallest_idx)) {
			feature.trackingIdx = i;
			feature.meshIdx = smallest_idx;
			feature.trackingPath = tracking_path;
			FeatureTrack.push_back(feature);

			nearestIdx.push_back(smallest_idx);
			tracking_path.clear();
		}
		else {
			tracking_path.push_back(i);
		}
	}

}

// Feature sketching Fuctions
void getFeatureSketch() {
	findAllNearestVertex();

	if (FeatureTrack.size() > 1) {
		featureTracking();
		SketchCorrespondence();

		getVerticesInPathTriangles();
		setFeaturePosition();
		get_RandomCp();

		FeaturePath.push_back(feature_path);
		num_feature += feature_path.size();
		//testFeatureSize();
	}
}

void DrawFeaturePath() {
	for (int i = 1; i < feature_path.size(); i++) {
		int idx = feature_path[i];
		int idx1 = feature_path[i - 1];
		glLineWidth(5.0f);
		glBegin(GL_LINES);
		glColor3f(0.0, 0.0, 1.0);
		glVertex3f(mesh->vertices[3 * (idx + 1) + 0], mesh->vertices[3 * (idx + 1) + 1], mesh->vertices[3 * (idx + 1) + 2]);
		glVertex3f(mesh->vertices[3 * (idx1 + 1) + 0], mesh->vertices[3 * (idx1 + 1) + 1], mesh->vertices[3 * (idx1 + 1) + 2]);
		glVertex3f(0.0, 0.0, 0.0);
		glEnd();
		/*		
		glLineWidth(5.0f);
		glBegin(GL_LINES);
		glColor3f(1.0, 0.0, 0.0);
		glVertex3f(correspond_sketch[i - 1][0], correspond_sketch[i - 1][1], correspond_sketch[i - 1][2]);
		glVertex3f(correspond_sketch[i][0], correspond_sketch[i][1], correspond_sketch[i][2]);
		glVertex3f(0.0, 0.0, 0.0);
		glEnd();
		*/
	}
}

void DrawFeature() {
	for (int i = 1; i<feature_tracking.size(); i++) {
		glLineWidth(5.0f);
		glBegin(GL_LINES);
		glColor3f(0.0, 1.0, 0.0);
		glVertex3f(feature_tracking[i][0], feature_tracking[i][1], feature_tracking[i][2]);
		glVertex3f(feature_tracking[i - 1][0], feature_tracking[i - 1][1], feature_tracking[i - 1][2]);
		glVertex3f(0.0, 0.0, 0.0);
		glEnd();
	}
}

// -------------------------------- Display functions --------------------------------
// ---------------------------------------------
// display related functions
void Reshape(int width, int height)
{
	int base = min(width, height);

	tbReshape(width, height);
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (GLdouble)width / (GLdouble)height, 1.0, 128.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -3.5);

	WindWidth = width;
	WindHeight = height;
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eyex, eyey, eyez,
		0.0, 0.0, 0.0,
		0.0, 1.0, 0.0);

	drawContour();

	//draw silhouette
	if (current_mode == DRAW_MODE && !draw_silhouette_flag&& mouse_tracking.size() > 1) {
		drawCorrespondence();
		DrawSilhouette();
	}

	if (draw_silhouette_flag)
		DrawFirstSilhouette();

	if (draw_feature_flag)
		DrawFeature();

	if (!draw_feature_flag && FeatureTrack.size()>0)
		DrawFeaturePath();

	glPushMatrix();
	//tbMatrix();

	// render solid model
	glEnable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glmDraw(mesh, GLM_SMOOTH);

	
	// render wire model
	if (display_wire_flag) {
		glPolygonOffset(1.0, 1.0);
		glEnable(GL_POLYGON_OFFSET_FILL);
		glLineWidth(1.0f);
		glColor3f(0.6, 0.0, 0.8);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glmDraw(mesh, GLM_SMOOTH);
	}

	// render handle points
	glPointSize(5.0);
	glEnable(GL_POINT_SMOOTH);
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	for (int handleIter = 0; handleIter<handles.size(); handleIter++)
	{
		if (handleIter==selected_handle_id)
			glColor3fv(colors[1]);
		else
			glColor3fv(colors[4]);

		//glColor3fv(colors[handleIter%colors.size()]);
		for (int vertIter = 0; vertIter<handles[handleIter].size(); vertIter++)
		{
			int idx = handles[handleIter][vertIter];
			glVertex3fv((float *)&mesh->vertices[3 * idx]);
		}
	}
	glEnd();

	glPopMatrix();

	glFlush();
	glutSwapBuffers();
}

// --------------------------------- Mouse functions -------------------------------
// ---------------------------------------------
// mouse related functions
vector3 Unprojection_silhouettes(vector2 _2Dpos, float depth)
{
	float Depth;
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	//tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	glReadPixels((int)_2Dpos.x, viewport[3] - (int)_2Dpos.y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &Depth);

	double X = _2Dpos.x;
	double Y = _2Dpos.y;
	double wpos[3] = { 0.0, 0.0, 0.0 };
	Depth = depth;
	gluUnProject(X, ((double)viewport[3] - Y), (double)Depth, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0], &wpos[1], &wpos[2]);

	return vector3(wpos[0], wpos[1], wpos[2]);
}

vector3 Unprojection(vector2 _2Dpos)
{
	float Depth;
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	//tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	glReadPixels((int)_2Dpos.x, viewport[3] - (int)_2Dpos.y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &Depth);

	double X = _2Dpos.x;
	double Y = _2Dpos.y;
	double wpos[3] = { 0.0, 0.0, 0.0 };

	feature_depth.push_back(Depth);
	gluUnProject(X, ((double)viewport[3] - Y), (double)Depth, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0], &wpos[1], &wpos[2]);

	return vector3(wpos[0], wpos[1], wpos[2]);
}

vector2 projection_helper(vector3 _3Dpos)
{
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	double wpos[3] = { 0.0, 0.0, 0.0 };
	gluProject(_3Dpos.x, _3Dpos.y, _3Dpos.z, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0], &wpos[1], &wpos[2]);

	return vector2(wpos[0], (double)viewport[3] - wpos[1]);
}

void mouse(int button, int state, int x, int y)
{
	//Because when drag the mouse, its projection will be effect
	//tbMouse(button, state, x, y);

	// select handle --------------------------------------------------------------------------------------------
	if (current_mode == SELECT_MODE && button == GLUT_RIGHT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			select_x = x;
			select_y = y;
		}
		else
		{
			vector<int> this_handle;
			vector<int> SilVertex_handle;
			vector<Edge> Silhouette_handle;

			// project all mesh vertices to current viewport
			for (int vertIter = 1; vertIter<mesh->numvertices + 1; vertIter++)
			{
				vector3 pt(mesh->vertices[3 * vertIter + 0], mesh->vertices[3 * vertIter + 1], mesh->vertices[3 * vertIter + 2]);
				vector2 pos = projection_helper(pt);

				// if the projection is inside the box specified by mouse click&drag, add it to current handle
				if (pos.x >= select_x && pos.y >= select_y && pos.x <= x && pos.y <= y)
				{
					this_handle.push_back(vertIter);
					handle_size++;
				}
			}

			handles.push_back(this_handle);
			SilVertex.push_back(SilVertex_handle);
			Silhouette.push_back(Silhouette_handle);

			unHandleVertex();
			if (allVertex_notInHandle.size()<num_vertices-1) {
				getHandlesTriangles();
				selected_handle_id = handles.size() - 1;
				findContour();
			}
		}
	}

	// Select the next handle to be "selected_handle_id" -----------------------------------------------
	if (current_mode == SELECT_MODE && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		if (handles.size() > 0) {
			// switch to the next handle
			if (selected_handle_id < handles.size() - 1)
				selected_handle_id++;
			else if (selected_handle_id == handles.size() - 1)
				selected_handle_id = 0;
			cout << "Yot select handle " << selected_handle_id << endl;
			// find contour of the next handle
			if (selected_handle_id != refind[0] || refind[1] == 0) {
				findContour();
			}
		}
	}

	// Find another contour -----------------------------------------------
	if (current_mode == FIND_CONTOUR && button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
		// increase the startpoint
		if (refind[1] < contourTriangle_size - 1) {
			refind[1]++;
			cout << "refind= " << refind[1] << endl;
			reFindContour(refind[1]);
		}
		else {
			refind[1] = 0;
		}
	}
	else if (current_mode == FIND_CONTOUR && button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
		// decrease the startpoint
		if (refind[1] > 0) {
			refind[1]--;
			cout << "refind= " << refind[1] << endl;
			reFindContour(refind[1]);
		}
		else {
			refind[1] = contourTriangle_size - 1;
		}
	}

	// draw silhouette -----------------------------------------------get_RandomCp
	if (current_mode == DRAW_MODE && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		cout << endl << "------------------------------- Start drawing silhouette" << endl;
		if (handles.size() > 0) {
			mouse_tracking.clear();
			mouse_tracking_original.clear();
			draw_silhouette_flag = true;
		}

	}
	else if (current_mode == DRAW_MODE && button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		draw_silhouette_flag = false;

		if (mouse_tracking.size() > 1 && handles.size() > 0) {
			getCorrespondence();
			inSilhouetteOrNot();
			getSolverIdx();
		}
		cout << "----------------------------------------------" << endl;
	}

	// deformation with MODE 1 -----------------------------------------------
	if (current_mode == DEFORM_MODE1 && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		if (mouse_tracking.size() > 1 && handles.size() > 0) {
			time_record.ResetTimer();
			cout << endl << "------------------------------- Start deformation with MODE1" << endl;
			iteration_times = 0;
			iteration = true;
			dis_smallest = 1;

			while (iteration && iteration_times < iteration_limit) {
				cout << "Iteration " << iteration_times << ", ";
				update_E();
				LeastSquareSolver();
				Reconstruction();
				iteration_times++;
				Display();
			}
			iteration = false;
			cout << "Times of Iteration = " << iteration_times << endl;
			cout << "----------------------------------------------" << endl;
		}
	}

	// deformation with MODE 2 -----------------------------------------------
	if (current_mode == DEFORM_MODE2 && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {

		if (mouse_tracking.size() > 1 && handles.size() > 0) {
			time_record.ResetTimer();
			cout << endl << "------------------------------- Start deformation with MODE2" << endl;
			LeastSquareSolver_v2();
			Reconstruction();
			updateDelta();

			LeastSquareSolver_v2();
			Reconstruction();
			getOriginalDelta();
			cout << "----------------------------------------------" << endl;
		}
	}

	// draw silhouette -----------------------------------------------
	if (current_mode == FEATURE_MODE && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		if (!deform_feature_flag) {
			//if click button feature sketch
			cout << endl << "------------------------------- Start drawing feature sketch" << endl;
			feature_tracking.clear();
			feature_depth.clear();
			draw_feature_flag = true;
			Weight_feature.weight = 10;
		}
	}
	else if (current_mode == FEATURE_MODE && button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		if (deform_feature_flag) {
			//if click button + and - in FEATURE_MODE
			deform_feature_flag = false;
			scaleCotDelta();
			LeastSquareSolver_update();
		}
		else if (feature_tracking.size() > 1) {
			//after click button feature sketch
			draw_feature_flag = false;
			getFeatureSketch();
			LeastSquareSolver_feature();
			getCot_Connectivity();
			getCot_Delta();
			//updateCotangent(&feature_roundVertices);
			cout << "----------------------------------------------" << endl;
		}
	}

	last_x = x;
	last_y = y;
}

void motion(int x, int y)
{
	tbMotion(x, y);

	if (current_mode == CLEAR_MODE) {
		BackToOriginal();
	}

	// if in draw mode and a handle is selected, deform the mesh
	if (current_mode == DRAW_MODE && draw_silhouette_flag == true) {
		vector3 vec3_buffer = Unprojection_silhouettes(vector2(x, y), depthOfSilhouette);
		mouse_tracking.push_back(vec3_buffer);
		mouse_tracking_original.push_back(vec3_buffer);
	}
	else if (current_mode == FEATURE_MODE && draw_feature_flag == true) {
		vector3 vec3_buffer = Unprojection(vector2(x, y));
		if (feature_depth[feature_depth.size() - 1]<1 && feature_depth[feature_depth.size() - 1]>0) {
			feature_tracking.push_back(vec3_buffer);
		}
		else {
			feature_depth.pop_back();
		}
	}

	last_x = x;
	last_y = y;
}

void moveEyes() {

	//findContour();
}

// -------------------------------- keyboard functions -----------------------------
// ---------------------------------------------
// keyboard related functions
void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	/* eyes look at -----------*/
	case 'd':
		eyex += 0.1;
		break;
	case 'a':
		eyex -= 0.1;
		break;
	case 'w':
		eyey += 0.1;
		break;
	case 's':
		eyey -= 0.1;
		break;
	case 'e':
		eyez += 0.1;
		break;
	case 'q':
		eyez -= 0.1;
		break;
	}
}

// ----------------------------------- main function -------------------------------
// ---------------------------------------------
void timf(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1, timf, 0);
}

int main(int argc, char *argv[])
{
	WindWidth = 800;
	WindHeight = 800;

	GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat light_diffuse[] = { 0.8, 0.8, 0.8, 1.0 };
	GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_position[] = { 0.0, 0.0, 1.0, 0.0 };

	// color list for rendering handles
	float red[] = { 1.0, 0.0, 0.0 };
	colors.push_back(red);
	float yellow[] = { 1.0, 0.7, 0.0 };
	colors.push_back(yellow);
	float blue[] = { 0.0, 1.0, 0.8 };
	colors.push_back(blue);
	float green[] = { 0.0, 1.0, 0.0 };
	colors.push_back(green);
	float pink[] = { 0.0, 0.8, 0.6 };
	colors.push_back(pink);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutTimerFunc(40, timf, 0); // Set up timer for 40ms, about 25 fps

	//------------------------------------------------------------------------------------------------------------------------------
	glutInitWindowPosition(600 - winw, 100);
	glutInitWindowSize(winw, winh);
	glutCreateWindow("UI");

	glutReshapeFunc(Resize);
	glutDisplayFunc(Draw);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
	glutPassiveMotionFunc(MousePassiveMotion);
	Init();
	//------------------------------------------------------------------------------------------------------------------------------

	glutInitWindowPosition(600, 100);
	glutInitWindowSize(WindWidth, WindHeight);
	glutCreateWindow("ARAP");

	glutReshapeFunc(Reshape);
	glutDisplayFunc(Display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);
	glClearColor(0, 0, 0, 0);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHT0);
	glDepthFunc(GL_LESS);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	tbInit(GLUT_LEFT_BUTTON);
	tbAnimate(GL_TRUE);

	// load 3D model : Armadillo, Dino, man, murphy
	//--------------------------------------------------------
	char str[] = "../data/";
	string s_buf;
	while (strcmp(filename, "Armadillo") && strcmp(filename, "Dino") && strcmp(filename, "murphy") && strcmp(filename, "cat") && strcmp(filename, "tiger"))
	{
		s_buf = filename;
		puts("Input the name of model below you want to deform ...");
		puts("Armadillo, Dino, murphy, cat or tiger.");
		scanf("%s", &filename);
	}
	strcat(str, filename);
	strcat(str, ".obj");
	s_buf = str;
	char *cstr = new char[s_buf.length() + 1];
	strcpy(cstr, s_buf.c_str());
	//--------------------------------------------------------

	mesh = glmReadOBJ(cstr);
	mesh_original = glmReadOBJ(cstr);

	glmUnitize(mesh);
	glmFacetNormals(mesh);
	glmVertexNormals(mesh, 90.0);

	glmUnitize(mesh_original);
	glmFacetNormals(mesh_original);
	glmVertexNormals(mesh_original, 90.0);

	ini_resize();
	findLongestLength();
	getUmbrella();
	get_E();
	get_triangles_position();
	getContangent();

	glutMainLoop();

	return 0;

}