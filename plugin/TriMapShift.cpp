#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/NukeWrapper.h"

using namespace DD::Image;

static const char* const CLASS = "TriMapShift";
static const char* const HELP = "Uses channel shifting to turn an existing mask into a triMap"; // The tooltip
                                                                                                //
class TriMapShift : public Iop {
	int value[3]; // Create user variable for per-channel translation

public:

	TriMapShift(Node* node) : Iop(node) { // Set all default values here
		value[0] = value[1] = value[2] = 0; // For instance, all items in the int value[3] should be 0 by default
	}

	void _validate(bool); // This will define the output image, like the size and channels it will have
	void _request(int x, int y, int r, int t, ChannelMask channels, int count); // This requests information from the input
	void engine(int y, int x, int r, ChannelMask, Row & row); // Where the calculations take place

	const char* Class() const { return d.name; }
	const char* node_help() const { return HELP; }

	virtual void knobs(Knob_Callback); // This is where knobs can be assigned
	static const Iop::Description d; // Make Nuke understand this node (at the bottom of the script more information on this)
};

void TriMapShift::_validate(bool for_real) {
	copy_info(); // Output image will have the same properties as the input
}

void TriMapShift::_request(int x, int y, int r, int t, ChannelMask channels, int count) {
	input0().request(x, y, r, t, channels, count); // Request the image data from input 0
}

void TriMapShift::engine(int y, int x, int r, ChannelMask channels, Row& row) {
	row.get(input0(), y, x, r, channels); // Fetch the data row

	foreach(z, channels) { // Do for each channel (0, 1, 2 = R, G, B)
		int rowlength = info().r(); // Get width of screen

		const int PixOffset = value[colourIndex(z)]; // Get current channel offset value for values 0, 1, 2. If higher than 2, pick value[0]

		const float* INP = row[z]; // Copy the pointer
		float* OUTP = row.writable(z); // Allocate the output pointer

		for (int X = x; X < r; X++) { // For each horizontal pixel within the row
			int NewPixel = int(X + PixOffset); // NewPixel = X + PixOffset

			if (!(NewPixel > 0 && NewPixel < rowlength)) { // Check the pixel index for illegal values
				NewPixel = -1; // When NewPixel does not exist within array INP, assign error value
			}

			float NewColor = (NewPixel == -1) ? 0 : INP[NewPixel]; // NewColor = INP value at pixel NewPixel, except if the value is -1
			OUTP[X] = NewColor; // Set OUTP at X to NewColour
		}
	}
}

void TriMapShift::knobs(Knob_Callback f) {
	MultiInt_knob(f, value, 3, "value"); // Make the user knob
	Tooltip(f, "Set the translation in R, G, and B."); // Set the tooltip for this knob
}

static Iop* build(Node *node) {
	return new NukeWrapper(new TriMapShift(node));
}
const Iop::Description TriMapShift::d("TriMapShift", "TriMapShift", build); // Nuke reads this to get the name of your plugin in the UI. Make sure this is the same as the name of the class and the final .dll file!
