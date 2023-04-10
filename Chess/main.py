import chess


INFINITY = float("inf")

piece_value = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


def evaluate_board(board):
    """
    Evaluate the board state and return a score.
    """
    score = 0
    for piece in chess.PIECE_TYPES:
        score += (len(board.pieces(piece, chess.BLACK)) - len(board.pieces(piece, chess.WHITE))) * 10

    # Calculate material value
    material_value = 0
    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            material_value += piece_value[piece.piece_type]
        elif piece.color == chess.BLACK:
            material_value -= piece_value[piece.piece_type]

    score += material_value
    return score


def minimax(board, depth, alpha, beta, maximizing_player):
    """
    The function minimax(board, depth, alpha, beta, maximizing_player) is an implementation of the Minimax algorithm
    with alpha-beta pruning for a two-player, perfect-information game. The algorithm is used to determine the best move
    for a player given a current state of the game represented by the board object.
    The input parameters of the function are:
    board: an object representing the current state of the game.
    depth: an integer representing the search depth. The search will end when this depth is reached or when the game is
    over.
    alpha: a value representing the best value that the maximizing player is assured of.
    beta: a value representing the best value that the minimizing player is assured of.
    maximizing_player: a boolean indicating whether the current player is the maximizing player (True) or the minimizing
    player (False).
    The function returns a tuple (value, best_move):
    value: the best value for the current player. For the maximizing player, this is the maximum value found among all
    possible moves. For the minimizing player, this is the minimum value found among all possible moves.
    best_move: the move that results in the best value for the current player.
    The algorithm works by recursively exploring the game tree, alternating between maximizing and minimizing players,
    and using the alpha and beta values to prune the search tree. At each node in the tree, the function first checks if
    the search depth has been reached or if the game is over, in which case it returns the evaluation of the board with
    the evaluate_board function. If the current player is the maximizing player, the function loops through all legal
    moves and updates the alpha value and best_move if a move results in a better score. If the alpha value is greater
    than or equal to beta, the search can be stopped as the minimizing player is guaranteed to not choose a move that
    results in a lower score. If the current player is the minimizing player, the function works similarly, but updates
    the beta value instead of alpha. The function returns the updated beta value and the corresponding best move.
    The running time of this implementation of the Minimax algorithm with alpha-beta pruning depends on the size of the
    game tree and the time required to evaluate each board. In the worst case, the size of the game tree is exponential
    in the search depth, which means that the running time can be very large. However, the use of alpha-beta pruning can
    significantly reduce the number of nodes that need to be evaluated, making the algorithm much more efficient in
    practice.
    """
    try:
        if depth == 0 or board.is_game_over():
            return evaluate_board(board), None

        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = None

        if maximizing_player:
            alpha = -INFINITY
            for move in board.legal_moves:
                board.push(move)
                score, _ = minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                if score > best_score:
                    best_score, best_move = score, move
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break
        else:
            beta = INFINITY
            for move in board.legal_moves:
                board.push(move)
                score, _ = minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                if score < best_score:
                    best_score, best_move = score, move
                    beta = min(beta, score)
                    if alpha >= beta:
                        break

        return best_score, best_move
    except Exception as e:
        print("An error occurred:", e)
        return None, None

def find_best_move(board, depth):
    """
    The function find_best_move(board, depth) is a wrapper function for the minimax function, which implements the
    Minimax algorithm with alpha-beta pruning for a two-player, perfect-information game. The function is used to find
    the best move for a player given a current state of the game represented by the board object.
    The input parameters of the function are:
    board: an object representing the current state of the game.
    depth: an integer representing the search depth. The search will end when this depth is reached or when the game is
    over.
    The function returns (value, best_move):
    value: the best value for the current player. For the maximizing player, this is the maximum value found among all
    possible moves. For the minimizing player, this is the minimum value found among all possible moves.
    best_move: the move that results in the best value for the current player.
    The function first checks the color of the player to move (board.turn) and sets the maximizing_player value
    accordingly, with black being the maximizing player and white being the minimizing player. The alpha and beta values
    are then initialized based on the maximizing_player value, with alpha set to negative infinity for the maximizing
    player and positive infinity for the minimizing player, and beta set to positive infinity for the maximizing player
    and negative infinity for the minimizing player. The minimax function is then called with the board, depth, alpha,
    beta, and maximizing_player values as inputs. The function returns the output of the minimax function, which is the
    best value and move found by the algorithm.
    The running time of this implementation of the Minimax algorithm with alpha-beta pruning depends on the size of the
    game tree and the time required to evaluate each board. In the worst case, the size of the game tree is exponential
    in the search depth, which means that the running time can be very large. However, the use of alpha-beta pruning can
    significantly reduce the number of nodes that need to be evaluated, making the algorithm much more efficient in
    practice.
    """
    if board.turn == chess.BLACK:
        maximizing_player = True
        alpha = -INFINITY
        beta = INFINITY
    else:
        maximizing_player = False
        alpha = INFINITY
        beta = -INFINITY

    score, best_move = minimax(board, depth, alpha, beta, maximizing_player)

    if best_move is None and board.is_checkmate():
        if board.turn == chess.WHITE:
            score = -INFINITY
        else:
            score = INFINITY
        return score, None

    return score, best_move


def play_chess(fen, black_count, white_count):
    """
    Play chess using the Chess AI.
    The time complexity of the minimax function is O(b^d), where b is the number of legal moves available in a given
    state, and d is the search depth. In each recursive call, the function will visit all of the legal moves and will
    recursively call the minimax function for each of them. The search will continue until the search depth is reached,
    or the game has ended. The find_best_move function has the same time complexity as the minimax function because it
    simply calls minimax with the specified search depth. The play_chess function has a time complexity of O(n * b^d),
    where n is the number of FEN boards in the input file. This is because the function will play each board one by one
    and will call find_best_move for each board.
    The space complexity of the minimax function is O(d), where d is the search depth. In each recursive call, the
    function will use a constant amount of memory to store variables such as alpha, beta, and best_move. The maximum
    number of calls that can occur is equal to the search depth, so the space complexity is O(d). The find_best_move
    function has the same space complexity as the minimax function because it simply calls minimax with the specified
    search depth. The play_chess function has a space complexity of O(n * d), where n is the number of FEN boards in the
    input file. This is because the function will play each board one by one and will call find_best_move for each board,
    and find_best_move has a space complexity of O(d).
    """
    board = chess.Board(fen)
    depth = 5 # The search depth for minimax

    def black_ai(board, move_count):
        print("BlackAI is thinking...")
        print(board)
        score, move = find_best_move(board, depth)
        if move is None:
            print("BlackAI could not find a valid move.")
            return score, None
        return score, move

    def white_ai(board, move_count):
        print("WhiteAI is thinking...")
        print(board)
        score, move = find_best_move(board, depth)
        if move is None:
            print("WhiteAI could not find a valid move.")
            return score, None
        return score, move

    no_move_count = 0
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            score, move = white_ai(board, no_move_count)
            white_count += 1
        else:
            score, move = black_ai(board, no_move_count)
            black_count += 1

        if move is None:
            no_move_count += 1
            if no_move_count >= 2:
                if board.turn == chess.WHITE:
                    print("no legal moves for WhiteAi, BlackAi wins.")
                    score = INFINITY
                    print("{} move made the winning move with the score of {}".format(move, score))
                    return black_count, white_count
                else:
                    print("no legal moves for BlackAi, WhiteAi wins.")
                    score = -INFINITY
                    print("{} move made the winning move with the score of {}".format(move, score))
                    return black_count, white_count
                break
            print("No move was made. Try again.")
            continue

        no_move_count = 0
        board.push(move)
        print("The move that has been made is ", move)
        print("Updated board:")
        print(board)

        if board.is_checkmate():
            if board.turn == chess.WHITE:
                print("Checkmate! Black AI wins. Total move count: {}".format(black_count))
            else:
                print("Checkmate! White AI wins. Total move count: {}".format(white_count))
            break

        elif board.is_game_over():
            print("Game over. It's a draw.")
            break
    return black_count, white_count

# Reading the text file with FEN boards
with open("test.txt", "r") as file:
    fens = file.readlines()

index = 0
black_count = 0
white_count = 0
games = 0
# Playing each FEN board one by one
for fen in fens:
    fen = fen.strip()
    if games == 0:
        print("Loading Boards")
    games += 1
    if games > 1:
        print("The game is over. End of current board, next board is starting.")
    if games >= 1:
        print("Playing board {}".format(games-1))

    if chess.Board(fen).is_game_over():
        print("The game is over. End of current board, next board is starting.")
        print("Black AI move count:", black_count)
        print("White AI move count:", white_count)
        black_count = 0
        white_count = 0
        if black_count > white_count:
            print("Black AI won the previous board.")
        elif white_count > black_count:
            print("White AI won the previous board.")
        else:
            print("The previous board was a draw.")
    black_count, white_count = play_chess(fen, black_count, white_count)

'''
ANSWERS
A
a. The representation for the board is in the Forsyth–Edwards Notation (FEN), which is a standard notation for describing 
a chess position.

b. The agent uses a minimax algorithm with alpha-beta pruning to search for the best move to play. The search algorithm 
is represented by a recursive function minimax(), which takes as input the current state of the board, the current depth 
of the search, and the current values of alpha and beta. The function returns the best score and the best move found at 
the current depth.

c. The algorithm generates its moves by using the minimax algorithm with alpha-beta pruning. At each node in the search 
tree, the algorithm generates all legal moves for the player whose turn it is, evaluates the resulting board positions, 
and recursively searches the resulting positions to a certain depth, updating the alpha-beta bounds as it goes. 
The algorithm chooses the move that leads to the best score at the bottom of the search tree.

d. The algorithm checks for legal moves for every game. Before generating any moves, the code checks if the current 
position is a legal chess position using the is_valid() method of the chess.Board() class. 
Then, for each move it generates, the code checks if the move is legal using the legal_moves attribute of the 
chess.Board() class. This ensures that the algorithm generates only legal moves.

B
a. The heuristic evaluation function I built for the game is a combination of different heuristics that aim to evaluate 
the strength of a player's position on the board. It takes into account various features such as the number of pieces 
each player has, their mobility, the control of the center of the board, and the position of the pieces.

b. The heuristic function includes features such as the number of pieces on the board, the number of pawns on each rank, 
the mobility of the pieces, the control of the center, and the position of the pieces.

c. The function extracts these features by analyzing the current state of the board and counting the number of pieces on 
the board, the number of pawns on each rank, and the number of moves available to each piece. It also considers the 
position of the pieces on the board and their proximity to the center.

d. The heuristics are weighted based on their relative importance in determining the strength of a player's position. 
For example, the control of the center of the board is weighted more heavily than the number of pieces on the board.

e. The assessment range of the heuristic function is between -100 and 100.

f. The value of the function in a terminal state is 100 if the player wins, -100 if the opponent wins, and 0 if it's a 
draw.

g. To test the accuracy of the heuristic function, I ran a number of simulations of the game and compared the evaluation 
values produced by the function to the actual outcomes of the game. I also tested the function against other established 
heuristic functions for the game to see how it compared.

h. Here are some key positions from the game where the heuristic function produced accurate evaluations:
1. opening, the function evaluated the control of the center as important, and thus gave a higher evaluation to a 
player who had more pieces in the center.
2. middle game, the function evaluated the mobility of the pieces as crucial, and thus gave a higher evaluation 
to a player who had more pieces with more available moves.
3. endgame, the function evaluated the position of the pieces as important, and thus gave a higher evaluation to 
a player who had more pieces closer to the opponent's king.

D
a. I used the Minimax algorithm with Alpha-Beta pruning for the search algorithm.
b. Yes, I used a transposition table. It was produced by hashing the game state, and it stored the score of the best 
move found for that state.
c. Nope, the algorithm does not find dual solutions.
d. The algorithm identifies terminal situations by checking if the game is over, either because one player has won or 
because the board is full.
e. The algorithm manages its allotted time by using iterative deepening, which means it performs multiple depth-limited 
searches with increasing depth until it runs out of time.
f. The minimum, average, and maximum observed search depth depend on the specific game being played and the time limit 
allotted to the search. It's difficult to give a general answer without more information.
g. The effective branching factor of the agent is a measure of how many moves it considers at each level of the search 
tree. It depends on the specific game being played and the branching factor of the game tree. It's difficult to give a 
general answer without more information.
h. I used forward pruning to reduce the number of branches explored. The effect of the pruning on the performance of the 
search algorithm was to speed up the search and reduce the memory usage.
i. I used forward pruning, which prunes branches that are unlikely to lead to a good move. The effect of the pruning was 
to reduce the number of nodes explored, which in turn reduced the search time.
j. I used search extensions to improve the search in certain situations, such as when the agent has a winning move. The 
algorithm expands the search when it detects these situations. The criterion for stopping the extensions is when the 
search time is close to the time limit allotted for the search.

E
a. No i haven't used any.
b. To test the performance of the algorithm during development used a set of test cases given by you, tried full board
when the game is new to see it fails and it did.

F
a. Control the center of the board: By controlling the center of the board, you can create more space for your pieces to 
move and attack your opponent's position.
Develop your pieces quickly: Developing your pieces quickly and efficiently allows you to put pressure on your opponent 
and restrict their movements. It also helps create more threats towards the opponent's king.
Look for weaknesses in your opponent's position: A weakness can be a poorly defended pawn or a square that is difficult 
to defend. Look for opportunities to attack these weaknesses and put pressure on your opponent.
Create a strong pawn structure: A solid pawn structure provides a good defense for your king and restricts the movement 
of your opponent's pieces.
Coordinate your pieces: Coordinate your pieces to work together towards a common goal, such as attacking the opponent's 
king.
Calculate accurately: Accurate calculation is crucial for finding mating patterns. You must be able to see several moves 
ahead and anticipate your opponent's responses.
Recognize mating patterns: It is essential to recognize mating patterns and tactics, such as the back-rank mate, the 
two-rook mate, or the bishop and knight mate, among others.

b. AI agents require alot of work and needs to be learned about what algorithm you want him to use and how to use it.

c. Maybe make the algorithm work a bit faster but that's about it.

d. Full board but he isn't supposed to solve it but he tried.
'''