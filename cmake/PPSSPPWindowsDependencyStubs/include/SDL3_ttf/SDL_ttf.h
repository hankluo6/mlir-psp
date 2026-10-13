#pragma once

// draw_text_sdl.h is parsed when USE_SDL3_TTF is globally defined, even
// though native Windows selects TextDrawerWin32/TextDrawerUWP.  Only this
// opaque declaration is needed on that code path.
typedef struct TTF_Font TTF_Font;
