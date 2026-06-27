class ANSI:
    # Reset
    RESET = "\033[0m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"

    # Standard foreground colours
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colours
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colours
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Bright background colours
    BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_RED = "\033[101m"
    BG_BRIGHT_GREEN = "\033[102m"
    BG_BRIGHT_YELLOW = "\033[103m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_MAGENTA = "\033[105m"
    BG_BRIGHT_CYAN = "\033[106m"
    BG_BRIGHT_WHITE = "\033[107m"

    @staticmethod
    def rgb(red, green, blue):
        return f"\033[38;2;{red};{green};{blue}m"

    @staticmethod
    def bgRgb(red, green, blue):
        return f"\033[48;2;{red};{green};{blue}m"
    
    def enumToRgb(enumValue):
        match enumValue:
            case ANSI.RED | ANSI.BG_RED:
                return (255, 0, 0)
            case ANSI.GREEN | ANSI.BG_GREEN:
                return (0, 255, 0)
            case ANSI.BLUE | ANSI.BG_BLUE:
                return (0, 0, 255)
            case ANSI.YELLOW | ANSI.BG_YELLOW:
                return (255, 255, 0)
            case ANSI.CYAN | ANSI.BG_CYAN:
                return (0, 255, 255)
            case ANSI.MAGENTA | ANSI.BG_MAGENTA:
                return (255, 0, 255)
            case ANSI.WHITE | ANSI.BG_WHITE:
                return (255, 255, 255)
            case ANSI.BLACK | ANSI.BG_BLACK:
                return (0, 0, 0)
            case _:
                return (255, 255, 255)  # Default to white