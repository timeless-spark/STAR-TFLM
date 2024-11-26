library ieee;
use ieee.std_logic_1164.all;

package BOUGH_WOOLEY_TYPES is
    type b4_Prop_type is array(integer range 0 to 3) of std_logic_vector(3 downto 0);
    type b4_Carry_type is array(integer range 0 to 2) of std_logic_vector(3 downto 0);
    type b4_Sum_type is array(integer range 0 to 2) of std_logic_vector(3 downto 1);
    type b16_Prop_type is array(integer range 0 to 15) of b4_Prop_type;
    type b16_Carry_type is array(integer range 0 to 2) of std_logic_vector(15 downto 0);
    type b16_Sum_record is record right: std_logic_vector(2 downto 0); diag: std_logic; down: std_logic_vector(2 downto 0); end record;
    type b16_Sum_type is array(integer range 0 to 15) of b16_Sum_record;
end package;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

entity HA is
    port 
    (
        A  : in  std_logic;
        B  : in  std_logic;
        S  : out std_logic;
        Co : out std_logic
    );
end entity;

architecture structural of HA is
begin

    S  <= A xor B;
    Co <= (A and B);
    
end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

entity FA is
    port 
    (
        A  : in  std_logic;
        B  : in  std_logic;
        Ci : in  std_logic;
        S  : out std_logic;
        Co : out std_logic
    );
end entity;

architecture structural of FA is
    begin
    
        S  <= A xor B xor Ci;
        Co <= (A and B) or (B and Ci) or (A and Ci);
        
end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

entity WHITE_BLOCK is
    port
    (
        Ai : in  std_logic;
        Bi : in  std_logic;
        Ci : in  std_logic; --carry in
        Pi : in  std_logic; --"propagate" selector
        Si : in  std_logic; --sum in
        So : out std_logic; --sum out
        Co : out std_logic  --carry out
    );
end entity;

architecture structural of WHITE_BLOCK is
    component FA is
        port
        (
            A  : in  std_logic;
            B  : in  std_logic;
            Ci : in  std_logic;
            S  : out std_logic;
            Co : out std_logic
        );
    end component;

    signal S_ab : std_logic;
    signal S_mux: std_logic;

    begin
        S_ab <= (Ai and Bi) and Pi;
        FAi : entity work.FA(structural) port map(A=>S_ab, B=>Si, Ci=>Ci, S=>So, Co=>Co);
    
end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

entity BLUE_BLOCK is
    port
    (
        Ai : in  std_logic;
        Bi : in  std_logic;
        Ci : in  std_logic; --carry in
        Ii : in  std_logic; --"sum 1" selector
        Pi : in  std_logic; --"propagate" selector
        Si : in  std_logic; --sum in
        So : out std_logic; --sum out
        Co : out std_logic  --carry out
    );
end entity;

architecture structural of BLUE_BLOCK is
    component FA is
        port
        (
            A  : in  std_logic;
            B  : in  std_logic;
            Ci : in  std_logic;
            S  : out std_logic;
            Co : out std_logic
        );
    end component;

    signal S_ab : std_logic;

    begin
        S_ab <= ((Ai and Bi) and Pi) or Ii;
        FAi : entity work.FA(structural) port map(A=>S_ab, B=>Si, Ci=>Ci, S=>So, Co=>Co);
    
end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- library ieee;
-- use ieee.std_logic_1164.all;

-- entity GREEN_BLOCK is
--     port
--     (
--         Ai : in  std_logic;
--         Bi : in  std_logic;
--         Ci : in  std_logic; --carry in
--         Ii : in  std_logic; --"sum 1" selector
--         Pi : in  std_logic; --"propagate" selector
--         Si : in  std_logic; --sum in
--         So : out std_logic; --sum out
--         Co : out std_logic  --carry out
--     );
-- end entity;

-- architecture structural of GREEN_BLOCK is
--     component FA is
--         port
--         (
--             A  : in  std_logic;
--             B  : in  std_logic;
--             Ci : in  std_logic;
--             S  : out std_logic;
--             Co : out std_logic
--         );
--     end component;

--     signal S_ab : std_logic;

--     begin
--         S_ab <= ((Ai nand Bi) and Pi) or Ii;
--         FAi : entity work.FA(structural) port map(A=>S_ab, B=>Si, Ci=>Ci, S=>So, Co=>Co);
    
-- end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

entity RED_BLOCK is
    port
    (
        Ai : in  std_logic;
        Bi : in  std_logic;
        Ci : in  std_logic; --carry in
        Ii : in  std_logic; --"sum 1" selector
        Pi : in  std_logic; --"propagate" selector
        Si : in  std_logic; --sum in
        So : out std_logic; --sum out
        Co : out std_logic  --carry out
    );
end entity;

architecture structural of RED_BLOCK is
    component FA is
        port
        (
            A  : in  std_logic;
            B  : in  std_logic;
            Ci : in  std_logic;
            S  : out std_logic;
            Co : out std_logic
        );
    end component;

    signal S_ab : std_logic;

    begin
        S_ab <= ((Ai and Bi) and Pi) xor Ii;
        FAi : entity work.FA(structural) port map(A=>S_ab, B=>Si, Ci=>Ci, S=>So, Co=>Co);
    
end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V00 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V00 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),          Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1),    Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2),    Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3),    Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),          Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1),    Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2),    Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3),    Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),          Co=>Co(0),       Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),          Co=>Co(1),       Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),          Co=>Co(2),       Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),          Co=>Co(3),       Pi=>Pi(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V01 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V01 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),       Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1), Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2), Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3), Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),       Pi=>Pi(3)(0));
        --position 3,1
        P31 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),       Pi=>Pi(3)(1));
        --position 3,2
        P32 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),       Pi=>Pi(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),       Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V02 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V02 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),       Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1), Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2), Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3), Co=>C_int(0)(3), Pi=>Pi(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int(1)(3), Pi=>Pi(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int(2)(3), Pi=>Pi(2)(3));
        --position 3,0
        P30 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),       Pi=>Pi(3)(0));
        --position 3,1
        P31 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),       Pi=>Pi(3)(1));
        --position 3,2
        P32 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),       Pi=>Pi(3)(2));
        --position 3,3
        P33 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),       Pi=>Pi(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V03 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V03 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),       Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1), Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2), Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3), Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),       Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),       Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),       Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),       Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V10 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Xi : in  std_logic;                    --'1' propagate carries through diagonal, '0' block propagation
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V10 is

    signal C_int_in: b4_Carry_type;
    signal C_int_out: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --carry-in connections
        C_int_in(0)(0) <= C_int_out(0)(0);
        C_int_in(0)(1) <= C_int_out(0)(1);
        C_int_in(0)(2) <= C_int_out(0)(2);
        C_int_in(0)(3) <= C_int_out(0)(3) and Xi;
        C_int_in(1)(0) <= C_int_out(1)(0);
        C_int_in(1)(1) <= C_int_out(1)(1);
        C_int_in(1)(2) <= C_int_out(1)(2) and Xi;
        C_int_in(1)(3) <= C_int_out(1)(3);
        C_int_in(2)(0) <= C_int_out(2)(0);
        C_int_in(2)(1) <= C_int_out(2)(1) and Xi;
        C_int_in(2)(2) <= C_int_out(2)(2);
        C_int_in(2)(3) <= C_int_out(2)(3);
    
        --position 0,0
        P00 : entity work.BLUE_BLOCK(structural)  port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int_out(0)(0), Pi=>Pi(0)(0), Ii=>Ii(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int_out(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int_out(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int_out(0)(3), Pi=>Pi(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int_in(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int_out(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int_in(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int_out(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int_in(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int_out(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int_in(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int_out(1)(3), Pi=>Pi(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int_in(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int_out(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int_in(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int_out(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int_in(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int_out(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int_in(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int_out(2)(3), Pi=>Pi(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int_in(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),           Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int_in(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),           Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int_in(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),           Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int_in(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),           Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V11 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V11 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),          Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1),    Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2),    Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3),    Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),          Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1),    Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2),    Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3),    Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),          Co=>Co(0),       Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),          Co=>Co(1),       Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),          Co=>Co(2),       Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),          Co=>Co(3),       Pi=>Pi(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V12 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V12 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3)  );
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),          Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1),    Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2),    Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3),    Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3)  );
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),          Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1),    Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2),    Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3),    Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3)  );
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),          Co=>Co(0),       Pi=>Pi(3)(0), Ii=>Ii(3)(0)  );
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),          Co=>Co(1),       Pi=>Pi(3)(1), Ii=>Ii(3)(1)  );
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),          Co=>Co(2),       Pi=>Pi(3)(2), Ii=>Ii(3)(2)  );
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),          Co=>Co(3),       Pi=>Pi(3)(3), Ii=>Ii(3)(3)  );

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V13 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V13 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),          Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1),    Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2),    Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3),    Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),          Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1),    Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2),    Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3),    Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),          Co=>Co(0),       Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),          Co=>Co(1),       Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),          Co=>Co(2),       Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),          Co=>Co(3),       Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V20 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V20 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.BLUE_BLOCK(structural)  port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),       Co=>C_int(0)(0), Pi=>Pi(0)(0), ii=>ii(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1), Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2), Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3), Co=>C_int(0)(3), Pi=>Pi(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int(1)(3), Pi=>Pi(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int(2)(3), Pi=>Pi(2)(3));
        --position 3,0
        P30 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),       Pi=>Pi(3)(0));
        --position 3,1
        P31 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),       Pi=>Pi(3)(1));
        --position 3,2
        P32 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),       Pi=>Pi(3)(2));
        --position 3,3
        P33 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),       Pi=>Pi(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V21 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Xi : in  std_logic;                    --'1' propagate carries through diagonal, '0' block propagation
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V21 is

    signal C_int_in: b4_Carry_type;
    signal C_int_out: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --carry-in connections
        C_int_in(0)(0) <= C_int_out(0)(0);
        C_int_in(0)(1) <= C_int_out(0)(1);
        C_int_in(0)(2) <= C_int_out(0)(2);
        C_int_in(0)(3) <= C_int_out(0)(3) and Xi;
        C_int_in(1)(0) <= C_int_out(1)(0);
        C_int_in(1)(1) <= C_int_out(1)(1);
        C_int_in(1)(2) <= C_int_out(1)(2) and Xi;
        C_int_in(1)(3) <= C_int_out(1)(3);
        C_int_in(2)(0) <= C_int_out(2)(0);
        C_int_in(2)(1) <= C_int_out(2)(1) and Xi;
        C_int_in(2)(2) <= C_int_out(2)(2);
        C_int_in(2)(3) <= C_int_out(2)(3);
    
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int_out(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int_out(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int_out(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int_out(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int_in(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int_out(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int_in(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int_out(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int_in(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int_out(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int_in(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int_out(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int_in(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int_out(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int_in(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int_out(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int_in(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int_out(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int_in(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int_out(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int_in(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),           Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int_in(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),           Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int_in(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),           Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int_in(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),           Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V22 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V22 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),       Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.BLUE_BLOCK(structural)  port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1), Co=>C_int(0)(1), Pi=>Pi(0)(1), Ii=>Ii(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2), Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3), Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),       Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),       Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),       Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),       Pi=>Pi(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V23 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V23 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),          Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1),    Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2),    Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3),    Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),          Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1),    Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2),    Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3),    Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),          Co=>Co(0),       Pi=>Pi(3)(0));
        --position 3,1
        P31 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),          Co=>Co(1),       Pi=>Pi(3)(1));
        --position 3,2
        P32 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),          Co=>Co(2),       Pi=>Pi(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),          Co=>Co(3),       Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V30 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Xi : in  std_logic;                    --'1' propagate carries through diagonal, '0' block propagation
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V30 is

    signal C_int_in: b4_Carry_type;
    signal C_int_out: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --carry-in connections
        C_int_in(0)(0) <= C_int_out(0)(0);
        C_int_in(0)(1) <= C_int_out(0)(1);
        C_int_in(0)(2) <= C_int_out(0)(2);
        C_int_in(0)(3) <= C_int_out(0)(3) and Xi;
        C_int_in(1)(0) <= C_int_out(1)(0);
        C_int_in(1)(1) <= C_int_out(1)(1);
        C_int_in(1)(2) <= C_int_out(1)(2) and Xi;
        C_int_in(1)(3) <= C_int_out(1)(3);
        C_int_in(2)(0) <= C_int_out(2)(0);
        C_int_in(2)(1) <= C_int_out(2)(1) and Xi;
        C_int_in(2)(2) <= C_int_out(2)(2);
        C_int_in(2)(3) <= C_int_out(2)(3);
    
        --position 0,0
        P00 : entity work.BLUE_BLOCK(structural)  port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int_out(0)(0), Pi=>Pi(0)(0), Ii=>Ii(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int_out(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int_out(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int_out(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int_in(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int_out(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int_in(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int_out(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int_in(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int_out(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int_in(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int_out(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int_in(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int_out(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int_in(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int_out(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int_in(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int_out(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int_in(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int_out(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int_in(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),           Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int_in(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),           Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int_in(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),           Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int_in(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),           Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V31 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V31 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),       Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1), Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.BLUE_BLOCK(structural)  port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2), Co=>C_int(0)(2), Pi=>Pi(0)(2), Ii=>Ii(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3), Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),       Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),       Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),       Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),       Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V32 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Xi : in  std_logic;                    --'1' propagate carries through diagonal, '0' block propagation
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V32 is

    signal C_int_in: b4_Carry_type;
    signal C_int_out: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --carry-in connections
        C_int_in(0)(0) <= C_int_out(0)(0);
        C_int_in(0)(1) <= C_int_out(0)(1);
        C_int_in(0)(2) <= C_int_out(0)(2);
        C_int_in(0)(3) <= C_int_out(0)(3) and Xi;
        C_int_in(1)(0) <= C_int_out(1)(0);
        C_int_in(1)(1) <= C_int_out(1)(1);
        C_int_in(1)(2) <= C_int_out(1)(2) and Xi;
        C_int_in(1)(3) <= C_int_out(1)(3);
        C_int_in(2)(0) <= C_int_out(2)(0);
        C_int_in(2)(1) <= C_int_out(2)(1) and Xi;
        C_int_in(2)(2) <= C_int_out(2)(2);
        C_int_in(2)(3) <= C_int_out(2)(3);
    
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int_out(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int_out(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int_out(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int_out(0)(3), Pi=>Pi(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int_in(0)(0), Si=>S_int(0)(1), So=>So(1),       Co=>C_int_out(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int_in(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1), Co=>C_int_out(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int_in(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2), Co=>C_int_out(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int_in(0)(3), Si=>Si(4),       So=>S_int(1)(3), Co=>C_int_out(1)(3), Pi=>Pi(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int_in(1)(0), Si=>S_int(1)(1), So=>So(2),       Co=>C_int_out(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int_in(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1), Co=>C_int_out(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int_in(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2), Co=>C_int_out(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int_in(1)(3), Si=>Si(5),       So=>S_int(2)(3), Co=>C_int_out(2)(3), Pi=>Pi(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int_in(2)(0), Si=>S_int(2)(1), So=>So(3),       Co=>Co(0),           Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int_in(2)(1), Si=>S_int(2)(2), So=>So(4),       Co=>Co(1),           Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int_in(2)(2), Si=>S_int(2)(3), So=>So(5),       Co=>Co(2),           Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int_in(2)(3), Si=>Si(6),       So=>So(6),       Co=>Co(3),           Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B4_BLOCK_V33 is
    port
    (
        Ai : in  std_logic_vector(3 downto 0);
        Bi : in  std_logic_vector(3 downto 0);
        Ci : in  std_logic_vector(3 downto 0); --carry in
        Pi : in  b4_Prop_type;                 --"propagate" selector
        Ii : in  b4_Prop_type;                 --"invert" selector
        Si : in  std_logic_vector(6 downto 0); --sum in
        So : out std_logic_vector(6 downto 0); --sum out
        Co : out std_logic_vector(3 downto 0)  --carry out
    );
end entity;

architecture structural of B4_BLOCK_V33 is

    signal C_int: b4_Carry_type;
    signal S_int: b4_Sum_type;

    begin
        --position 0,0
        P00 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(0), Ci=>Ci(0),       Si=>Si(0),       So=>So(0),          Co=>C_int(0)(0), Pi=>Pi(0)(0));
        --position 0,1
        P01 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(0), Ci=>Ci(1),       Si=>Si(1),       So=>S_int(0)(1),    Co=>C_int(0)(1), Pi=>Pi(0)(1));
        --position 0,2
        P02 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(0), Ci=>Ci(2),       Si=>Si(2),       So=>S_int(0)(2),    Co=>C_int(0)(2), Pi=>Pi(0)(2));
        --position 0,3
        P03 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(0), Ci=>Ci(3),       Si=>Si(3),       So=>S_int(0)(3),    Co=>C_int(0)(3), Pi=>Pi(0)(3), Ii=>Ii(0)(3));
        --position 1,0
        P10 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(1), Ci=>C_int(0)(0), Si=>S_int(0)(1), So=>So(1),          Co=>C_int(1)(0), Pi=>Pi(1)(0));
        --position 1,1
        P11 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(1), Ci=>C_int(0)(1), Si=>S_int(0)(2), So=>S_int(1)(1),    Co=>C_int(1)(1), Pi=>Pi(1)(1));
        --position 1,2
        P12 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(1), Ci=>C_int(0)(2), Si=>S_int(0)(3), So=>S_int(1)(2),    Co=>C_int(1)(2), Pi=>Pi(1)(2));
        --position 1,3
        P13 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(1), Ci=>C_int(0)(3), Si=>Si(4),       So=>S_int(1)(3),    Co=>C_int(1)(3), Pi=>Pi(1)(3), Ii=>Ii(1)(3));
        --position 2,0
        P20 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(0), Bi=>Bi(2), Ci=>C_int(1)(0), Si=>S_int(1)(1), So=>So(2),          Co=>C_int(2)(0), Pi=>Pi(2)(0));
        --position 2,1
        P21 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(1), Bi=>Bi(2), Ci=>C_int(1)(1), Si=>S_int(1)(2), So=>S_int(2)(1),    Co=>C_int(2)(1), Pi=>Pi(2)(1));
        --position 2,2
        P22 : entity work.WHITE_BLOCK(structural) port map(Ai=>Ai(2), Bi=>Bi(2), Ci=>C_int(1)(2), Si=>S_int(1)(3), So=>S_int(2)(2),    Co=>C_int(2)(2), Pi=>Pi(2)(2));
        --position 2,3
        P23 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(2), Ci=>C_int(1)(3), Si=>Si(5),       So=>S_int(2)(3),    Co=>C_int(2)(3), Pi=>Pi(2)(3), Ii=>Ii(2)(3));
        --position 3,0
        P30 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(0), Bi=>Bi(3), Ci=>C_int(2)(0), Si=>S_int(2)(1), So=>So(3),          Co=>Co(0),       Pi=>Pi(3)(0), Ii=>Ii(3)(0));
        --position 3,1
        P31 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(1), Bi=>Bi(3), Ci=>C_int(2)(1), Si=>S_int(2)(2), So=>So(4),          Co=>Co(1),       Pi=>Pi(3)(1), Ii=>Ii(3)(1));
        --position 3,2
        P32 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(2), Bi=>Bi(3), Ci=>C_int(2)(2), Si=>S_int(2)(3), So=>So(5),          Co=>Co(2),       Pi=>Pi(3)(2), Ii=>Ii(3)(2));
        --position 3,3
        P33 : entity work.RED_BLOCK(structural)   port map(Ai=>Ai(3), Bi=>Bi(3), Ci=>C_int(2)(3), Si=>Si(6),       So=>So(6),          Co=>Co(3),       Pi=>Pi(3)(3), Ii=>Ii(3)(3));

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity B16_BLOCK_V0 is
    port
    (
        Ai : in  std_logic_vector(15 downto 0);
        Bi : in  std_logic_vector(15 downto 0);
        Ci : in  std_logic_vector(15 downto 0); --carry in
        Pi : in  b16_Prop_type;                 --"propagate" selector
        Ii : in  b16_Prop_type;                 --"invert" selector
        Xi : in  std_logic_vector(2 downto 0);  --'1' propagate carries through diagonal, '0' block propagation
        Si : in  std_logic_vector(30 downto 0); --sum in
        So : out std_logic_vector(30 downto 0); --sum out
        Co : out std_logic_vector(15 downto 0)  --carry out
    );
end entity;

architecture structural of B16_BLOCK_V0 is

    signal C_int: b16_Carry_type;
    signal C_int_out: std_logic_vector(3 downto 0);
    signal S_int: b16_Sum_type;

    begin
        --position 0,0
        B00: entity work.B4_BLOCK_V00(structural) port map(Ai=>Ai(3 downto 0),   Bi=>Bi(3 downto 0),   Ci=>Ci(3 downto 0),         Ii=>Ii(0),  Pi=>Pi(0),  Si(6 downto 4)=>S_int(1).right,   Si(3)=>Si(3),          Si(2 downto 0)=>Si(2 downto 0),   So(6 downto 4)=>S_int(0).down,    So(3)=>So(3),          So(2 downto 0)=>So(2 downto 0),   Co=>C_int(0)(3 downto 0));
        --position 0,1
        B01: entity work.B4_BLOCK_V01(structural) port map(Ai=>Ai(7 downto 4),   Bi=>Bi(3 downto 0),   Ci=>Ci(7 downto 4),         Ii=>Ii(1),  Pi=>Pi(1),  Si(6 downto 4)=>S_int(2).right,   Si(3)=>Si(7),          Si(2 downto 0)=>Si(6 downto 4),   So(6 downto 4)=>S_int(1).down,    So(3)=>S_int(1).diag,  So(2 downto 0)=>S_int(1).right,   Co=>C_int(0)(7 downto 4));
        --position 0,2
        B02: entity work.B4_BLOCK_V02(structural) port map(Ai=>Ai(11 downto 8),  Bi=>Bi(3 downto 0),   Ci=>Ci(11 downto 8),        Ii=>Ii(2),  Pi=>Pi(2),  Si(6 downto 4)=>S_int(3).right,   Si(3)=>Si(11),         Si(2 downto 0)=>Si(10 downto 8),  So(6 downto 4)=>S_int(2).down,    So(3)=>S_int(2).diag,  So(2 downto 0)=>S_int(2).right,   Co=>C_int(0)(11 downto 8));
        --position 0,3
        B03: entity work.B4_BLOCK_V03(structural) port map(Ai=>Ai(15 downto 12), Bi=>Bi(3 downto 0),   Ci=>Ci(15 downto 12),       Ii=>Ii(3),  Pi=>Pi(3),  Si(6 downto 4)=>Si(18 downto 16), Si(3)=>Si(15),         Si(2 downto 0)=>Si(14 downto 12), So(6 downto 4)=>S_int(3).down,    So(3)=>S_int(3).diag,  So(2 downto 0)=>S_int(3).right,   Co=>C_int(0)(15 downto 12));
        --position 1,0
        B10: entity work.B4_BLOCK_V10(structural) port map(Ai=>Ai(3 downto 0),   Bi=>Bi(7 downto 4),   Ci=>C_int(0)(3 downto 0),   Ii=>Ii(4),  Pi=>Pi(4),  Si(6 downto 4)=>S_int(5).right,   Si(3)=>S_int(1).diag,  Si(2 downto 0)=>S_int(0).down,    So(6 downto 4)=>S_int(4).down,    So(3)=>So(7),          So(2 downto 0)=>So(6 downto 4),   Co(3 downto 1)=>C_int(1)(3 downto 1), Co(0)=>C_int_out(0), Xi=>Xi(0));
        --position 1,1
        B11: entity work.B4_BLOCK_V11(structural) port map(Ai=>Ai(7 downto 4),   Bi=>Bi(7 downto 4),   Ci=>C_int(0)(7 downto 4),   Ii=>Ii(5),  Pi=>Pi(5),  Si(6 downto 4)=>S_int(6).right,   Si(3)=>S_int(2).diag,  Si(2 downto 0)=>S_int(1).down,    So(6 downto 4)=>S_int(5).down,    So(3)=>S_int(5).diag,  So(2 downto 0)=>S_int(5).right,   Co=>C_int(1)(7 downto 4));
        --position 1,2
        B12: entity work.B4_BLOCK_V12(structural) port map(Ai=>Ai(11 downto 8),  Bi=>Bi(7 downto 4),   Ci=>C_int(0)(11 downto 8),  Ii=>Ii(6),  Pi=>Pi(6),  Si(6 downto 4)=>S_int(7).right,   Si(3)=>S_int(3).diag,  Si(2 downto 0)=>S_int(2).down,    So(6 downto 4)=>S_int(6).down,    So(3)=>S_int(6).diag,  So(2 downto 0)=>S_int(6).right,   Co=>C_int(1)(11 downto 8));
        --position 1,3
        B13: entity work.B4_BLOCK_V13(structural) port map(Ai=>Ai(15 downto 12), Bi=>Bi(7 downto 4),   Ci=>C_int(0)(15 downto 12), Ii=>Ii(7),  Pi=>Pi(7),  Si(6 downto 4)=>Si(22 downto 20), Si(3)=>Si(19),         Si(2 downto 0)=>S_int(3).down,    So(6 downto 4)=>S_int(7).down,    So(3)=>S_int(7).diag,  So(2 downto 0)=>S_int(7).right,   Co=>C_int(1)(15 downto 12));
        --position 2,0
        B20: entity work.B4_BLOCK_V20(structural) port map(Ai=>Ai(3 downto 0),   Bi=>Bi(11 downto 8),  Ci=>C_int(1)(3 downto 0),   Ii=>Ii(8),  Pi=>Pi(8),  Si(6 downto 4)=>S_int(9).right,   Si(3)=>S_int(5).diag,  Si(2 downto 0)=>S_int(4).down,    So(6 downto 4)=>S_int(8).down,    So(3)=>So(11),         So(2 downto 0)=>So(10 downto 8),  Co=>C_int(2)(3 downto 0));  
        --position 2,1
        B21: entity work.B4_BLOCK_V21(structural) port map(Ai=>Ai(7 downto 4),   Bi=>Bi(11 downto 8),  Ci=>C_int(1)(7 downto 4),   Ii=>Ii(9),  Pi=>Pi(9),  Si(6 downto 4)=>S_int(10).right,  Si(3)=>S_int(6).diag,  Si(2 downto 0)=>S_int(5).down,    So(6 downto 4)=>S_int(9).down,    So(3)=>S_int(9).diag,  So(2 downto 0)=>S_int(9).right,   Co(3 downto 1)=>C_int(2)(7 downto 5), Co(0)=>C_int_out(1), Xi=>Xi(1));
        --position 2,2
        B22: entity work.B4_BLOCK_V22(structural) port map(Ai=>Ai(11 downto 8),  Bi=>Bi(11 downto 8),  Ci=>C_int(1)(11 downto 8),  Ii=>Ii(10), Pi=>Pi(10), Si(6 downto 4)=>S_int(11).right,  Si(3)=>S_int(7).diag,  Si(2 downto 0)=>S_int(6).down,    So(6 downto 4)=>S_int(10).down,   So(3)=>S_int(10).diag, So(2 downto 0)=>S_int(10).right,  Co=>C_int(2)(11 downto 8));
        --position 2,3
        B23: entity work.B4_BLOCK_V23(structural) port map(Ai=>Ai(15 downto 12), Bi=>Bi(11 downto 8),  Ci=>C_int(1)(15 downto 12), Ii=>Ii(11), Pi=>Pi(11), Si(6 downto 4)=>Si(26 downto 24), Si(3)=>Si(23),         Si(2 downto 0)=>S_int(7).down,    So(6 downto 4)=>S_int(11).down,   So(3)=>S_int(11).diag, So(2 downto 0)=>S_int(11).right,  Co=>C_int(2)(15 downto 12));
        --position 3,0
        B30: entity work.B4_BLOCK_V30(structural) port map(Ai=>Ai(3 downto 0),   Bi=>Bi(15 downto 12), Ci=>C_int(2)(3 downto 0),   Ii=>Ii(12), Pi=>Pi(12), Si(6 downto 4)=>S_int(13).right,  Si(3)=>S_int(9).diag,  Si(2 downto 0)=>S_int(8).down,    So(6 downto 4)=>So(18 downto 16), So(3)=>So(15),         So(2 downto 0)=>So(14 downto 12), Co(3 downto 1)=>Co(3 downto 1), Co(0)=>C_int_out(2), Xi=>Xi(1));
        --position 3,1
        B31: entity work.B4_BLOCK_V31(structural) port map(Ai=>Ai(7 downto 4),   Bi=>Bi(15 downto 12), Ci=>C_int(2)(7 downto 4),   Ii=>Ii(13), Pi=>Pi(13), Si(6 downto 4)=>S_int(14).right,  Si(3)=>S_int(10).diag, Si(2 downto 0)=>S_int(9).down,    So(6 downto 4)=>So(22 downto 20), So(3)=>So(19),         So(2 downto 0)=>S_int(13).right,  Co=>Co(7 downto 4));
        --position 3,2
        B32: entity work.B4_BLOCK_V32(structural) port map(Ai=>Ai(11 downto 8),  Bi=>Bi(15 downto 12), Ci=>C_int(2)(11 downto 8),  Ii=>Ii(14), Pi=>Pi(14), Si(6 downto 4)=>S_int(15).right,  Si(3)=>S_int(11).diag, Si(2 downto 0)=>S_int(10).down,   So(6 downto 4)=>So(26 downto 24), So(3)=>So(23),         So(2 downto 0)=>S_int(14).right,  Co(3 downto 1)=>Co(11 downto 9), Co(0)=>C_int_out(3), Xi=>Xi(2));
        --position 3,3
        B33: entity work.B4_BLOCK_V33(structural) port map(Ai=>Ai(15 downto 12), Bi=>Bi(15 downto 12), Ci=>C_int(2)(15 downto 12), Ii=>Ii(15), Pi=>Pi(15), Si(6 downto 4)=>Si(30 downto 28), Si(3)=>Si(27),         Si(2 downto 0)=>S_int(11).down,   So(6 downto 4)=>So(30 downto 28), So(3)=>So(27),         So(2 downto 0)=>S_int(15).right,  Co=>Co(15 downto 12));

        --carry-out connections
        C_int(1)(0)  <= C_int_out(0) and Xi(0);
        C_int(2)(4)  <= C_int_out(1) and Xi(1);
        Co(0)        <= C_int_out(2) and Xi(1);
        Co(8)        <= C_int_out(3) and Xi(2);

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity RCA is
    generic 
    (
        N : integer := 4
    );
    port
    (
        A  : in  std_logic_vector(N-1 downto 0);
        B  : in  std_logic_vector(N-1 downto 0);
        Ci : in  std_logic;
        S  : out std_logic_vector(N-1 downto 0);
        Co : out std_logic
    );
end entity;

architecture structural of RCA is 

    component FA is
        port 
        (
            A  : in  std_logic;
            B  : in  std_logic;
            Ci : in  std_logic;
            S  : out std_logic;
            Co : out std_logic
        );
    end component;

    signal int_carry: std_logic_vector(N downto 0);

    begin
        int_carry(0) <= Ci;

        ADDER: for i in 0 to N-1 generate
            FAi: FA
            port map (A(i), B(i), int_carry(i), S(i), int_carry(i+1));
        end generate;

        Co <= int_carry(N);
end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.math_real.all;

--separable inteso a passo di 8 bit..
entity ADDER_16bit_SEP is
    port 
    (
        A  : in  std_logic_vector(15 downto 0);
        B  : in  std_logic_vector(15 downto 0);
        Ci : in  std_logic;
        Psel : in std_logic;                     --"propagate" selector  Psel = '1' -> normal behavior
        S  : out std_logic_vector(15 downto 0);  --                      Psel = '0' -> separated
        Co : out std_logic
    );
end entity;

architecture RCA of ADDER_16bit_SEP is

    component RCA is
        generic 
        (
            N : integer := 8
        );
        port
        (
            A  : in  std_logic_vector(N-1 downto 0);
            B  : in  std_logic_vector(N-1 downto 0);
            Ci : in  std_logic;
            S  : out std_logic_vector(N-1 downto 0);
            Co : out std_logic
        );
    end component;

    signal int_carry: std_logic;
    signal masked_carry: std_logic;
    
    begin

        masked_carry <= int_carry and Psel;

        RCA0: RCA port map(A=>A(7 downto 0),   B=>B(7 downto 0), Ci=>Ci,           S=>S(7 downto 0), Co=>int_carry);
        RCA1: RCA port map(A=>A(15 downto 8), B=>B(15 downto 8), Ci=>masked_carry, S=>S(15 downto 8), Co=>Co);

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity MUL16x16_CTRL is
    port 
    (
        CTRL_ARR   : in  std_logic_vector(25 downto 0);
        
        BW_Pi_ctrl : out b16_Prop_type;
        BW_Ii_ctrl : out b16_Prop_type
    );
end entity;

architecture structural of MUL16x16_CTRL is

    begin

      --assign each function to the proper block in the BW_MUL

      --prop_sel
        --(0,0)
        BW_Pi_ctrl(0)(0)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(0)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(0)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(0)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(1)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(1)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(1)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(1)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(2)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(2)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(2)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(2)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(3)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(3)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(3)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(0)(3)(3) <= CTRL_ARR(0); -- a
        --(0,1)
        BW_Pi_ctrl(1)(0)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(0)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(0)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(0)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(1)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(1)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(1)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(1)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(2)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(2)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(2)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(2)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(3)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(3)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(3)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(1)(3)(3) <= CTRL_ARR(1); -- b
        --(0,2)
        BW_Pi_ctrl(2)(0)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(0)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(0)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(0)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(1)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(1)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(1)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(1)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(2)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(2)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(2)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(2)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(3)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(3)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(3)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(2)(3)(3) <= CTRL_ARR(2); -- c
        --(0,3)
        BW_Pi_ctrl(3)(0)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(0)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(0)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(0)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(1)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(1)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(1)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(1)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(2)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(2)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(2)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(2)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(3)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(3)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(3)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(3)(3)(3) <= CTRL_ARR(3); -- d
        --(1,0)
        BW_Pi_ctrl(4)(0)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(0)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(0)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(0)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(1)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(1)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(1)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(1)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(2)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(2)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(2)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(2)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(3)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(3)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(3)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(4)(3)(3) <= CTRL_ARR(1); -- b
        --(1,1)
        BW_Pi_ctrl(5)(0)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(0)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(0)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(0)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(1)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(1)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(1)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(1)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(2)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(2)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(2)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(2)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(3)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(3)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(3)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(5)(3)(3) <= CTRL_ARR(0); -- a
        --(1,2)
        BW_Pi_ctrl(6)(0)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(0)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(0)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(0)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(1)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(1)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(1)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(1)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(2)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(2)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(2)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(2)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(3)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(3)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(3)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(6)(3)(3) <= CTRL_ARR(3); -- d
        --(1,3)
        BW_Pi_ctrl(7)(0)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(0)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(0)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(0)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(1)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(1)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(1)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(1)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(2)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(2)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(2)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(2)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(3)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(3)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(3)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(7)(3)(3) <= CTRL_ARR(2); -- c
        --(2,0)
        BW_Pi_ctrl(8)(0)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(0)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(0)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(0)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(1)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(1)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(1)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(1)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(2)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(2)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(2)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(2)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(3)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(3)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(3)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(8)(3)(3) <= CTRL_ARR(2); -- c
        --(2,1)
        BW_Pi_ctrl(9)(0)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(0)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(0)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(0)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(1)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(1)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(1)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(1)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(2)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(2)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(2)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(2)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(3)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(3)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(3)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(9)(3)(3) <= CTRL_ARR(3); -- d
        --(2,2)
        BW_Pi_ctrl(10)(0)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(0)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(0)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(0)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(1)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(1)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(1)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(1)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(2)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(2)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(2)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(2)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(3)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(3)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(3)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(10)(3)(3) <= CTRL_ARR(0); -- a
        --(2,3)
        BW_Pi_ctrl(11)(0)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(0)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(0)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(0)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(1)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(1)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(1)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(1)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(2)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(2)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(2)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(2)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(3)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(3)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(3)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(11)(3)(3) <= CTRL_ARR(1); -- b
        --(3,0)
        BW_Pi_ctrl(12)(0)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(0)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(0)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(0)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(1)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(1)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(1)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(1)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(2)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(2)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(2)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(2)(3) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(3)(0) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(3)(1) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(3)(2) <= CTRL_ARR(3); -- d
        BW_Pi_ctrl(12)(3)(3) <= CTRL_ARR(3); -- d
        --(3,1)
        BW_Pi_ctrl(13)(0)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(0)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(0)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(0)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(1)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(1)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(1)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(1)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(2)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(2)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(2)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(2)(3) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(3)(0) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(3)(1) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(3)(2) <= CTRL_ARR(2); -- c
        BW_Pi_ctrl(13)(3)(3) <= CTRL_ARR(2); -- c
        --(3,2)
        BW_Pi_ctrl(14)(0)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(0)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(0)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(0)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(1)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(1)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(1)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(1)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(2)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(2)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(2)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(2)(3) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(3)(0) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(3)(1) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(3)(2) <= CTRL_ARR(1); -- b
        BW_Pi_ctrl(14)(3)(3) <= CTRL_ARR(1); -- b
        --(3,3)
        BW_Pi_ctrl(15)(0)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(0)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(0)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(0)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(1)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(1)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(1)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(1)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(2)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(2)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(2)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(2)(3) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(3)(0) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(3)(1) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(3)(2) <= CTRL_ARR(0); -- a
        BW_Pi_ctrl(15)(3)(3) <= CTRL_ARR(0); -- a
        
    --sign_sel
        --(0,0)
        BW_Ii_ctrl(0)(0)(0) <= '0';
        BW_Ii_ctrl(0)(0)(1) <= '0';
        BW_Ii_ctrl(0)(0)(2) <= '0';
        BW_Ii_ctrl(0)(0)(3) <= CTRL_ARR(4);
        BW_Ii_ctrl(0)(1)(0) <= '0';
        BW_Ii_ctrl(0)(1)(1) <= '0';
        BW_Ii_ctrl(0)(1)(2) <= '0';
        BW_Ii_ctrl(0)(1)(3) <= CTRL_ARR(4);
        BW_Ii_ctrl(0)(2)(0) <= '0';
        BW_Ii_ctrl(0)(2)(1) <= '0';
        BW_Ii_ctrl(0)(2)(2) <= '0';
        BW_Ii_ctrl(0)(2)(3) <= CTRL_ARR(4);
        BW_Ii_ctrl(0)(3)(0) <= CTRL_ARR(4);
        BW_Ii_ctrl(0)(3)(1) <= CTRL_ARR(4);
        BW_Ii_ctrl(0)(3)(2) <= CTRL_ARR(4);
        BW_Ii_ctrl(0)(3)(3) <= '0';
        --(0,1)
        BW_Ii_ctrl(1)(0)(0) <= '0';
        BW_Ii_ctrl(1)(0)(1) <= '0';
        BW_Ii_ctrl(1)(0)(2) <= '0';
        BW_Ii_ctrl(1)(0)(3) <= CTRL_ARR(5);
        BW_Ii_ctrl(1)(1)(0) <= '0';
        BW_Ii_ctrl(1)(1)(1) <= '0';
        BW_Ii_ctrl(1)(1)(2) <= '0';
        BW_Ii_ctrl(1)(1)(3) <= CTRL_ARR(5);
        BW_Ii_ctrl(1)(2)(0) <= '0';
        BW_Ii_ctrl(1)(2)(1) <= '0';
        BW_Ii_ctrl(1)(2)(2) <= '0';
        BW_Ii_ctrl(1)(2)(3) <= CTRL_ARR(5);
        BW_Ii_ctrl(1)(3)(0) <= '0';
        BW_Ii_ctrl(1)(3)(1) <= '0';
        BW_Ii_ctrl(1)(3)(2) <= '0';
        BW_Ii_ctrl(1)(3)(3) <= CTRL_ARR(5);
        --(0,2)
        BW_Ii_ctrl(2)(0)(0) <= '0';
        BW_Ii_ctrl(2)(0)(1) <= '0';
        BW_Ii_ctrl(2)(0)(2) <= '0';
        BW_Ii_ctrl(2)(0)(3) <= '0';
        BW_Ii_ctrl(2)(1)(0) <= '0';
        BW_Ii_ctrl(2)(1)(1) <= '0';
        BW_Ii_ctrl(2)(1)(2) <= '0';
        BW_Ii_ctrl(2)(1)(3) <= '0';
        BW_Ii_ctrl(2)(2)(0) <= '0';
        BW_Ii_ctrl(2)(2)(1) <= '0';
        BW_Ii_ctrl(2)(2)(2) <= '0';
        BW_Ii_ctrl(2)(2)(3) <= '0';
        BW_Ii_ctrl(2)(3)(0) <= '0';
        BW_Ii_ctrl(2)(3)(1) <= '0';
        BW_Ii_ctrl(2)(3)(2) <= '0';
        BW_Ii_ctrl(2)(3)(3) <= '0';
        --(0,3)
        BW_Ii_ctrl(3)(0)(0) <= '0';
        BW_Ii_ctrl(3)(0)(1) <= '0';
        BW_Ii_ctrl(3)(0)(2) <= '0';
        BW_Ii_ctrl(3)(0)(3) <= CTRL_ARR(6);
        BW_Ii_ctrl(3)(1)(0) <= '0';
        BW_Ii_ctrl(3)(1)(1) <= '0';
        BW_Ii_ctrl(3)(1)(2) <= '0';
        BW_Ii_ctrl(3)(1)(3) <= CTRL_ARR(6);
        BW_Ii_ctrl(3)(2)(0) <= '0';
        BW_Ii_ctrl(3)(2)(1) <= '0';
        BW_Ii_ctrl(3)(2)(2) <= '0';
        BW_Ii_ctrl(3)(2)(3) <= CTRL_ARR(6);
        BW_Ii_ctrl(3)(3)(0) <= CTRL_ARR(7);
        BW_Ii_ctrl(3)(3)(1) <= CTRL_ARR(7);
        BW_Ii_ctrl(3)(3)(2) <= CTRL_ARR(7);
        BW_Ii_ctrl(3)(3)(3) <= CTRL_ARR(8);
        --(1,0)
        BW_Ii_ctrl(4)(0)(0) <= CTRL_ARR(4);
        BW_Ii_ctrl(4)(0)(1) <= '0';
        BW_Ii_ctrl(4)(0)(2) <= '0';
        BW_Ii_ctrl(4)(0)(3) <= '0';
        BW_Ii_ctrl(4)(1)(0) <= '0';
        BW_Ii_ctrl(4)(1)(1) <= '0';
        BW_Ii_ctrl(4)(1)(2) <= '0';
        BW_Ii_ctrl(4)(1)(3) <= '0';
        BW_Ii_ctrl(4)(2)(0) <= '0';
        BW_Ii_ctrl(4)(2)(1) <= '0';
        BW_Ii_ctrl(4)(2)(2) <= '0';
        BW_Ii_ctrl(4)(2)(3) <= '0';
        BW_Ii_ctrl(4)(3)(0) <= CTRL_ARR(9);
        BW_Ii_ctrl(4)(3)(1) <= CTRL_ARR(5);
        BW_Ii_ctrl(4)(3)(2) <= CTRL_ARR(5);
        BW_Ii_ctrl(4)(3)(3) <= CTRL_ARR(5);
        --(1,1)
        BW_Ii_ctrl(5)(0)(0) <= '0';
        BW_Ii_ctrl(5)(0)(1) <= '0';
        BW_Ii_ctrl(5)(0)(2) <= '0';
        BW_Ii_ctrl(5)(0)(3) <= CTRL_ARR(9);
        BW_Ii_ctrl(5)(1)(0) <= '0';
        BW_Ii_ctrl(5)(1)(1) <= '0';
        BW_Ii_ctrl(5)(1)(2) <= '0';
        BW_Ii_ctrl(5)(1)(3) <= CTRL_ARR(9);
        BW_Ii_ctrl(5)(2)(0) <= '0';
        BW_Ii_ctrl(5)(2)(1) <= '0';
        BW_Ii_ctrl(5)(2)(2) <= '0';
        BW_Ii_ctrl(5)(2)(3) <= CTRL_ARR(9);
        BW_Ii_ctrl(5)(3)(0) <= CTRL_ARR(9);
        BW_Ii_ctrl(5)(3)(1) <= CTRL_ARR(9);
        BW_Ii_ctrl(5)(3)(2) <= CTRL_ARR(9);
        BW_Ii_ctrl(5)(3)(3) <= '0';
        --(1,2)
        BW_Ii_ctrl(6)(0)(0) <= '0';
        BW_Ii_ctrl(6)(0)(1) <= '0';
        BW_Ii_ctrl(6)(0)(2) <= '0';
        BW_Ii_ctrl(6)(0)(3) <= CTRL_ARR(7);
        BW_Ii_ctrl(6)(1)(0) <= '0';
        BW_Ii_ctrl(6)(1)(1) <= '0';
        BW_Ii_ctrl(6)(1)(2) <= '0';
        BW_Ii_ctrl(6)(1)(3) <= CTRL_ARR(7);
        BW_Ii_ctrl(6)(2)(0) <= '0';
        BW_Ii_ctrl(6)(2)(1) <= '0';
        BW_Ii_ctrl(6)(2)(2) <= '0';
        BW_Ii_ctrl(6)(2)(3) <= CTRL_ARR(7);
        BW_Ii_ctrl(6)(3)(0) <= CTRL_ARR(10);
        BW_Ii_ctrl(6)(3)(1) <= CTRL_ARR(10);
        BW_Ii_ctrl(6)(3)(2) <= CTRL_ARR(10);
        BW_Ii_ctrl(6)(3)(3) <= CTRL_ARR(11);
        --(1,3)
        BW_Ii_ctrl(7)(0)(0) <= '0';
        BW_Ii_ctrl(7)(0)(1) <= '0';
        BW_Ii_ctrl(7)(0)(2) <= '0';
        BW_Ii_ctrl(7)(0)(3) <= CTRL_ARR(8);
        BW_Ii_ctrl(7)(1)(0) <= '0';
        BW_Ii_ctrl(7)(1)(1) <= '0';
        BW_Ii_ctrl(7)(1)(2) <= '0';
        BW_Ii_ctrl(7)(1)(3) <= CTRL_ARR(8);
        BW_Ii_ctrl(7)(2)(0) <= '0';
        BW_Ii_ctrl(7)(2)(1) <= '0';
        BW_Ii_ctrl(7)(2)(2) <= '0';
        BW_Ii_ctrl(7)(2)(3) <= CTRL_ARR(8);
        BW_Ii_ctrl(7)(3)(0) <= CTRL_ARR(11);
        BW_Ii_ctrl(7)(3)(1) <= CTRL_ARR(11);
        BW_Ii_ctrl(7)(3)(2) <= CTRL_ARR(11);
        BW_Ii_ctrl(7)(3)(3) <= CTRL_ARR(12);
        --(2,0)
        BW_Ii_ctrl(8)(0)(0) <= CTRL_ARR(5);
        BW_Ii_ctrl(8)(0)(1) <= '0';
        BW_Ii_ctrl(8)(0)(2) <= '0';
        BW_Ii_ctrl(8)(0)(3) <= '0';
        BW_Ii_ctrl(8)(1)(0) <= '0';
        BW_Ii_ctrl(8)(1)(1) <= '0';
        BW_Ii_ctrl(8)(1)(2) <= '0';
        BW_Ii_ctrl(8)(1)(3) <= '0';
        BW_Ii_ctrl(8)(2)(0) <= '0';
        BW_Ii_ctrl(8)(2)(1) <= '0';
        BW_Ii_ctrl(8)(2)(2) <= '0';
        BW_Ii_ctrl(8)(2)(3) <= '0';
        BW_Ii_ctrl(8)(3)(0) <= '0';
        BW_Ii_ctrl(8)(3)(1) <= '0';
        BW_Ii_ctrl(8)(3)(2) <= '0';
        BW_Ii_ctrl(8)(3)(3) <= '0';
        --(2,1)
        BW_Ii_ctrl(9)(0)(0) <= '0';
        BW_Ii_ctrl(9)(0)(1) <= '0';
        BW_Ii_ctrl(9)(0)(2) <= '0';
        BW_Ii_ctrl(9)(0)(3) <= CTRL_ARR(10);
        BW_Ii_ctrl(9)(1)(0) <= '0';
        BW_Ii_ctrl(9)(1)(1) <= '0';
        BW_Ii_ctrl(9)(1)(2) <= '0';
        BW_Ii_ctrl(9)(1)(3) <= CTRL_ARR(10);
        BW_Ii_ctrl(9)(2)(0) <= '0';
        BW_Ii_ctrl(9)(2)(1) <= '0';
        BW_Ii_ctrl(9)(2)(2) <= '0';
        BW_Ii_ctrl(9)(2)(3) <= CTRL_ARR(10);
        BW_Ii_ctrl(9)(3)(0) <= CTRL_ARR(7);
        BW_Ii_ctrl(9)(3)(1) <= CTRL_ARR(7);
        BW_Ii_ctrl(9)(3)(2) <= CTRL_ARR(7);
        BW_Ii_ctrl(9)(3)(3) <= CTRL_ARR(11);
        --(2,2)
        BW_Ii_ctrl(10)(0)(0) <= '0';
        BW_Ii_ctrl(10)(0)(1) <= CTRL_ARR(11);
        BW_Ii_ctrl(10)(0)(2) <= '0';
        BW_Ii_ctrl(10)(0)(3) <= CTRL_ARR(4);
        BW_Ii_ctrl(10)(1)(0) <= '0';
        BW_Ii_ctrl(10)(1)(1) <= '0';
        BW_Ii_ctrl(10)(1)(2) <= '0';
        BW_Ii_ctrl(10)(1)(3) <= CTRL_ARR(4);
        BW_Ii_ctrl(10)(2)(0) <= '0';
        BW_Ii_ctrl(10)(2)(1) <= '0';
        BW_Ii_ctrl(10)(2)(2) <= '0';
        BW_Ii_ctrl(10)(2)(3) <= CTRL_ARR(4);
        BW_Ii_ctrl(10)(3)(0) <= CTRL_ARR(4);
        BW_Ii_ctrl(10)(3)(1) <= CTRL_ARR(4);
        BW_Ii_ctrl(10)(3)(2) <= CTRL_ARR(4);
        BW_Ii_ctrl(10)(3)(3) <= '0';
        --(2,3)
        BW_Ii_ctrl(11)(0)(0) <= '0';
        BW_Ii_ctrl(11)(0)(1) <= '0';
        BW_Ii_ctrl(11)(0)(2) <= '0';
        BW_Ii_ctrl(11)(0)(3) <= CTRL_ARR(13);
        BW_Ii_ctrl(11)(1)(0) <= '0';
        BW_Ii_ctrl(11)(1)(1) <= '0';
        BW_Ii_ctrl(11)(1)(2) <= '0';
        BW_Ii_ctrl(11)(1)(3) <= CTRL_ARR(13);
        BW_Ii_ctrl(11)(2)(0) <= '0';
        BW_Ii_ctrl(11)(2)(1) <= '0';
        BW_Ii_ctrl(11)(2)(2) <= '0';
        BW_Ii_ctrl(11)(2)(3) <= CTRL_ARR(13);
        BW_Ii_ctrl(11)(3)(0) <= '0';
        BW_Ii_ctrl(11)(3)(1) <= '0';
        BW_Ii_ctrl(11)(3)(2) <= '0';
        BW_Ii_ctrl(11)(3)(3) <= CTRL_ARR(13);
        --(3,0)
        BW_Ii_ctrl(12)(0)(0) <= CTRL_ARR(4);
        BW_Ii_ctrl(12)(0)(1) <= '0';
        BW_Ii_ctrl(12)(0)(2) <= '0';
        BW_Ii_ctrl(12)(0)(3) <= CTRL_ARR(7);
        BW_Ii_ctrl(12)(1)(0) <= '0';
        BW_Ii_ctrl(12)(1)(1) <= '0';
        BW_Ii_ctrl(12)(1)(2) <= '0';
        BW_Ii_ctrl(12)(1)(3) <= CTRL_ARR(7);
        BW_Ii_ctrl(12)(2)(0) <= '0';
        BW_Ii_ctrl(12)(2)(1) <= '0';
        BW_Ii_ctrl(12)(2)(2) <= '0';
        BW_Ii_ctrl(12)(2)(3) <= CTRL_ARR(7);
        BW_Ii_ctrl(12)(3)(0) <= CTRL_ARR(15);
        BW_Ii_ctrl(12)(3)(1) <= CTRL_ARR(16);
        BW_Ii_ctrl(12)(3)(2) <= CTRL_ARR(16);
        BW_Ii_ctrl(12)(3)(3) <= CTRL_ARR(17);
        --(3,1)
        BW_Ii_ctrl(13)(0)(0) <= '0';
        BW_Ii_ctrl(13)(0)(1) <= '0';
        BW_Ii_ctrl(13)(0)(2) <= CTRL_ARR(7);
        BW_Ii_ctrl(13)(0)(3) <= CTRL_ARR(11);
        BW_Ii_ctrl(13)(1)(0) <= '0';
        BW_Ii_ctrl(13)(1)(1) <= '0';
        BW_Ii_ctrl(13)(1)(2) <= '0';
        BW_Ii_ctrl(13)(1)(3) <= CTRL_ARR(11);
        BW_Ii_ctrl(13)(2)(0) <= '0';
        BW_Ii_ctrl(13)(2)(1) <= '0';
        BW_Ii_ctrl(13)(2)(2) <= '0';
        BW_Ii_ctrl(13)(2)(3) <= CTRL_ARR(11);
        BW_Ii_ctrl(13)(3)(0) <= CTRL_ARR(17);
        BW_Ii_ctrl(13)(3)(1) <= CTRL_ARR(18);
        BW_Ii_ctrl(13)(3)(2) <= CTRL_ARR(16);
        BW_Ii_ctrl(13)(3)(3) <= CTRL_ARR(19);
        --(3,2)
        BW_Ii_ctrl(14)(0)(0) <= '0';
        BW_Ii_ctrl(14)(0)(1) <= '0';
        BW_Ii_ctrl(14)(0)(2) <= '0';
        BW_Ii_ctrl(14)(0)(3) <= '0';
        BW_Ii_ctrl(14)(1)(0) <= '0';
        BW_Ii_ctrl(14)(1)(1) <= '0';
        BW_Ii_ctrl(14)(1)(2) <= '0';
        BW_Ii_ctrl(14)(1)(3) <= '0';
        BW_Ii_ctrl(14)(2)(0) <= '0';
        BW_Ii_ctrl(14)(2)(1) <= '0';
        BW_Ii_ctrl(14)(2)(2) <= '0';
        BW_Ii_ctrl(14)(2)(3) <= '0';
        BW_Ii_ctrl(14)(3)(0) <= CTRL_ARR(20);
        BW_Ii_ctrl(14)(3)(1) <= CTRL_ARR(21);
        BW_Ii_ctrl(14)(3)(2) <= CTRL_ARR(21);
        BW_Ii_ctrl(14)(3)(3) <= CTRL_ARR(21);
        --(3,3)
        BW_Ii_ctrl(15)(0)(0) <= '0';
        BW_Ii_ctrl(15)(0)(1) <= '0';
        BW_Ii_ctrl(15)(0)(2) <= '0';
        BW_Ii_ctrl(15)(0)(3) <= CTRL_ARR(14);
        BW_Ii_ctrl(15)(1)(0) <= '0';
        BW_Ii_ctrl(15)(1)(1) <= '0';
        BW_Ii_ctrl(15)(1)(2) <= '0';
        BW_Ii_ctrl(15)(1)(3) <= CTRL_ARR(14);
        BW_Ii_ctrl(15)(2)(0) <= '0';
        BW_Ii_ctrl(15)(2)(1) <= '0';
        BW_Ii_ctrl(15)(2)(2) <= '0';
        BW_Ii_ctrl(15)(2)(3) <= CTRL_ARR(14);
        BW_Ii_ctrl(15)(3)(0) <= CTRL_ARR(20);
        BW_Ii_ctrl(15)(3)(1) <= CTRL_ARR(20);
        BW_Ii_ctrl(15)(3)(2) <= CTRL_ARR(20);
        BW_Ii_ctrl(15)(3)(3) <= CTRL_ARR(23);

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity STAR_16x16_BW is
    port 
    (
      A, B : in  std_logic_vector(15 downto 0);
      CTRL : in  std_logic_vector(2 downto 0);
      P    : out std_logic_vector(31 downto 0)
    );
end entity;

architecture structural of STAR_16x16_BW is

    signal int_Pi: b16_Prop_type;
    signal int_Ii: b16_Prop_type;

    signal CTRL_ARR: std_logic_vector(25 downto 0);

    signal ctrl_a: std_logic;
    signal ctrl_b: std_logic;
    signal ctrl_c: std_logic;
    signal ctrl_d: std_logic;
    signal ctrl_e: std_logic;
    signal ctrl_f: std_logic;
    signal ctrl_g: std_logic;
    signal ctrl_h: std_logic;
    signal ctrl_i: std_logic;
    signal ctrl_j: std_logic;
    signal ctrl_k: std_logic;
    signal ctrl_l: std_logic;
    signal ctrl_m: std_logic;
    signal ctrl_n: std_logic;
    signal ctrl_o: std_logic;
    signal ctrl_p: std_logic;
    signal ctrl_q: std_logic;
    signal ctrl_r: std_logic;
    signal ctrl_s: std_logic;
    signal ctrl_t: std_logic;
    signal ctrl_u: std_logic;
    signal ctrl_v: std_logic;
    signal ctrl_w: std_logic;
    signal ctrl_x: std_logic;
    signal ctrl_y: std_logic;
    signal ctrl_z: std_logic;

    signal int_Ci: std_logic_vector(16 downto 0);
    signal int_Si: std_logic_vector(31 downto 0);
    signal int_So: std_logic_vector(14 downto 0);
    signal int_Co: std_logic_vector(15 downto 0);
    signal int_Xi: std_logic_vector(2 downto 0);

    signal int_out: std_logic_vector(31 downto 0);

    signal int_A, int_B: std_logic_vector(15 downto 0);

    signal sh_out: std_logic_vector(31 downto 0);

    begin
        -- to BW ppm
        ctrl_a <= ((not CTRL(2))) or (CTRL(0) and CTRL(1) and CTRL(2)) or (CTRL(0) and (not CTRL(1)) and CTRL(2));
        ctrl_b <= ((not CTRL(2))) or (CTRL(0) and (not CTRL(1)) and CTRL(2));
        ctrl_c <= ((not CTRL(2))) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2));
        ctrl_d <= ((not CTRL(2))) or ((not CTRL(0)) and CTRL(1) and CTRL(2)) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2));
        ctrl_e <= (CTRL(0) and CTRL(1) and CTRL(2));
        ctrl_f <= (CTRL(0) and (not CTRL(1)) and CTRL(2));
        ctrl_g <= (CTRL(0) and (not CTRL(1)) and (not CTRL(2))) or ((not CTRL(0)) and CTRL(1) and CTRL(2)) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_h <= ((not CTRL(0)) and CTRL(1) and CTRL(2));
        ctrl_i <= (CTRL(0) and (not CTRL(1)) and (not CTRL(2))) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_j <= (CTRL(0) and CTRL(1) and CTRL(2)) or (CTRL(0) and (not CTRL(1)) and CTRL(2));
        ctrl_k <= ((not CTRL(0)) and CTRL(1) and CTRL(2)) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2));
        ctrl_l <= ((not CTRL(0)) and (not CTRL(1)) and CTRL(2));
        ctrl_m <= (CTRL(0) and (not CTRL(1)) and (not CTRL(2))) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_n <= (CTRL(0) and (not CTRL(1)) and (not CTRL(2))) or (CTRL(0) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_o <= (CTRL(0) and (not CTRL(1)) and (not CTRL(2))) or (CTRL(0) and CTRL(1) and CTRL(2)) or (CTRL(0) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_p <= ((not CTRL(0)) and CTRL(1) and CTRL(2)) or (CTRL(0) and CTRL(1) and CTRL(2)) or ((not CTRL(0)) and CTRL(1) and (not CTRL(2))) or (CTRL(0) and (not CTRL(1)) and CTRL(2)) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_q <= ((not CTRL(0)) and CTRL(1) and CTRL(2)) or ((not CTRL(0)) and CTRL(1) and (not CTRL(2))) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_r <= ((not CTRL(0)) and CTRL(1) and (not CTRL(2))) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_s <= (CTRL(0) and CTRL(1) and CTRL(2)) or ((not CTRL(0)) and CTRL(1) and (not CTRL(2))) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_t <= ((not CTRL(0)) and CTRL(1) and (not CTRL(2))) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_u <= (CTRL(0) and CTRL(1) and CTRL(2)) or ((not CTRL(0)) and CTRL(1) and (not CTRL(2))) or (CTRL(0) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_v <= ((not CTRL(0)) and CTRL(1) and (not CTRL(2))) or (CTRL(0) and (not CTRL(1)) and CTRL(2)) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_w <= ((not CTRL(2))) or ((not CTRL(0)) and CTRL(1) and CTRL(2)) or (CTRL(0) and (not CTRL(1)) and CTRL(2)) or ((not CTRL(0)) and (not CTRL(1)) and CTRL(2));
        ctrl_x <= (CTRL(0) and (not CTRL(1)) and (not CTRL(2))) or ((not CTRL(0)) and CTRL(1) and (not CTRL(2)));
        ctrl_y <= (CTRL(0) and CTRL(1) and (not CTRL(2)));
        ctrl_z <= (CTRL(0) and (not CTRL(1)) and (not CTRL(2))) or (CTRL(0) and CTRL(1) and CTRL(2)) or ((not CTRL(0)) and CTRL(1) and (not CTRL(2))) or (CTRL(0) and (not CTRL(1)) and CTRL(2)) or (CTRL(0) and CTRL(1) and (not CTRL(2)));

        int_Xi(0) <= ctrl_w;
        int_Xi(1) <= ctrl_d;
        int_Xi(2) <= ctrl_w;

        int_Si(23 downto 0) <= (others => '0');
        int_Si(24) <= ctrl_f;
        int_Si(27 downto 25) <= (others => '0');
        int_Si(28) <= ctrl_e;
        int_Si(30 downto 29) <= (others => '0');
        int_Si(31) <= ctrl_z;

        int_Ci(14 downto 0) <= (others => '0');
        int_Ci(15) <= ctrl_x;
        int_Ci(16) <= ctrl_y;

        CTRL_ARR <= ctrl_z & ctrl_y & ctrl_x & ctrl_w & ctrl_v & ctrl_u & ctrl_t & ctrl_s & ctrl_r & ctrl_q & ctrl_p & ctrl_o & ctrl_n & ctrl_m & ctrl_l & ctrl_k & ctrl_j & ctrl_i & ctrl_h & ctrl_g & ctrl_f & ctrl_e & ctrl_d & ctrl_c & ctrl_b & ctrl_a;

        BW_MUL: entity work.B16_BLOCK_V0(structural) port map (Ai=>int_A, Bi=>int_B, Ci=>int_Ci(15 downto 0), Ii=>int_Ii, Pi=>int_Pi, Xi=>int_Xi, Si=>int_Si(30 downto 0), So(15 downto 0)=>int_out(15 downto 0), So(30 downto 16)=>int_So, Co=>int_Co);
        
        PP_ADD: entity work.ADDER_16bit_SEP(RCA) port map (A(15)=>int_Si(31), A(14 downto 0)=>int_So, B=>int_Co, Ci=>int_Ci(16), Psel=>int_Xi(2), S=>int_out(31 downto 16));
        
        CTRL_UNIT: entity work.MUL16x16_CTRL(structural) port map(CTRL_ARR=>CTRL_ARR, BW_Pi_ctrl=>int_Pi, BW_Ii_ctrl=>int_Ii);

        int_A <= A;
        int_B <= B;

        process(int_out, CTRL)
        begin
            sh_out <= int_out;
            if CTRL(2) = '1' then -- STAR
                if CTRL(0) = '0' then -- ST
                    if CTRL(1) = '0' then -- 8b
                        sh_out(15 downto 0) <= int_out(23 downto 8);
                        sh_out(31 downto 16) <= (others => int_out(24));
                    else -- 4b
                        sh_out(8 downto 0) <= int_out(20 downto 12);
                        sh_out(31 downto 9) <= (others => int_out(21));
                    end if;
                end if;                    
            end if;
        end process;

        P <= sh_out;

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity EXTENDED_ADDER is
    port 
    (
        A  : in  std_logic_vector(51 downto 0);
        B  : in  std_logic_vector(51 downto 0);

        Ctrl_mux0 : in std_logic;
        Ctrl_mux1 : in std_logic;
        Ctrl_mux2 : in std_logic;
        Psel : in std_logic_vector(2 downto 0); --"propagate" selector  Psel = '1' -> normal behavior
                                                --                      Psel = '0' -> separated
        S  : out std_logic_vector(51 downto 0)
    );
end entity;

architecture RCA of EXTENDED_ADDER is

    component RCA is
        generic 
        (
            N : integer
        );
        port
        (
            A  : in  std_logic_vector(N-1 downto 0);
            B  : in  std_logic_vector(N-1 downto 0);
            Ci : in  std_logic;
            S  : out std_logic_vector(N-1 downto 0);
            Co : out std_logic
        );
    end component;

    signal Ci, Co : std_logic_vector(7 downto 0);

    begin

        ADD_0: entity work.RCA(structural) generic map (N=>8) port map (A=>A( 7 downto  0), B=>B( 7 downto  0), Ci=>Ci(0), S=>S( 7 downto  0), Co=>Co(0));
        ADD_1: entity work.RCA(structural) generic map (N=>8) port map (A=>A(15 downto  8), B=>B(15 downto  8), Ci=>Ci(1), S=>S(15 downto  8), Co=>Co(1));
        ADD_2: entity work.RCA(structural) generic map (N=>8) port map (A=>A(23 downto 16), B=>B(23 downto 16), Ci=>Ci(2), S=>S(23 downto 16), Co=>Co(2));
        ADD_3: entity work.RCA(structural) generic map (N=>8) port map (A=>A(31 downto 24), B=>B(31 downto 24), Ci=>Ci(3), S=>S(31 downto 24), Co=>Co(3));

        ADD_4: entity work.RCA(structural) generic map (N=>5) port map (A=>A(36 downto 32), B=>B(36 downto 32), Ci=>Ci(4), S=>S(36 downto 32), Co=>Co(4));
        ADD_5: entity work.RCA(structural) generic map (N=>5) port map (A=>A(41 downto 37), B=>B(41 downto 37), Ci=>Ci(5), S=>S(41 downto 37), Co=>Co(5));
        ADD_6: entity work.RCA(structural) generic map (N=>5) port map (A=>A(46 downto 42), B=>B(46 downto 42), Ci=>Ci(6), S=>S(46 downto 42), Co=>Co(6));
        ADD_7: entity work.RCA(structural) generic map (N=>5) port map (A=>A(51 downto 47), B=>B(51 downto 47), Ci=>Ci(7), S=>S(51 downto 47), Co=>Co(7));

        Ci(0) <= '0';
        Ci(1) <= Psel(0) AND Co(0);
        Ci(2) <= Psel(1) AND Co(1);
        Ci(3) <= Psel(2) AND Co(2);
        Ci(4) <= Co(3);
        Ci(5) <= Co(4) when Ctrl_mux0 = '0' else Co(2);
        Ci(6) <= Co(5) when Ctrl_mux1 = '0' else Co(1);
        Ci(7) <= Co(6) when Ctrl_mux2 = '0' else Co(0);

end architecture;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.BOUGH_WOOLEY_TYPES.all;

entity IBEX_STAR_MUL is
    port 
    (
      A, B : in  std_logic_vector(31 downto 0);

      ALU_REGi : in  std_logic_vector(33 downto 0);
      ALU_REGo : out std_logic_vector(33 downto 0);

      MAC_REGi : in  std_logic_vector(103 downto 0);
      MAC_REGo : out std_logic_vector(103 downto 0);
      
      CTRL : in  std_logic_vector(8 downto 0);
      
      RESo    : out std_logic_vector(31 downto 0)
    );
end entity;

architecture behavioral of IBEX_STAR_MUL is

    component STAR_16x16_BW is
        port 
        (
          A, B : in  std_logic_vector(15 downto 0);
          CTRL : in  std_logic_vector(2 downto 0);
          P    : out std_logic_vector(31 downto 0)
        );
    end component;

    component EXTENDED_ADDER is
        port 
        (
            A  : in  std_logic_vector(51 downto 0);
            B  : in  std_logic_vector(51 downto 0);
    
            Ctrl_mux0 : in std_logic;
            Ctrl_mux1 : in std_logic;
            Ctrl_mux2 : in std_logic;
            Psel : in std_logic_vector(2 downto 0); --"propagate" selector  Psel = '1' -> normal behavior
                                                    --                      Psel = '0' -> separated
            S  : out std_logic_vector(51 downto 0)
        );
    end component;

    signal mul_A, mul_B : std_logic_vector(15 downto 0);
    signal mul_O : std_logic_vector(31 downto 0);
    signal mul_ctrl : std_logic_vector(2 downto 0);

    signal add_A, add_B : std_logic_vector(51 downto 0);
    signal add_O : std_logic_vector(51 downto 0);
    signal add_ctrl_mux0 : std_logic;
    signal add_ctrl_mux1 : std_logic;
    signal add_ctrl_mux2 : std_logic;
    signal add_Psel : std_logic_vector(2 downto 0);

    signal from_reg : std_logic_vector(31 downto 0);

    begin
        BW_MUL: entity work.STAR_16x16_BW(structural) port map (A=>mul_A, B=>mul_B, CTRL=>mul_ctrl, P=>mul_O);

        EXT_ADD: entity work.EXTENDED_ADDER(RCA) port map (A=>add_A, B=>add_B, Ctrl_mux0=>add_ctrl_mux0, Ctrl_mux1=>add_ctrl_mux1, Ctrl_mux2=>add_ctrl_mux2, Psel=>add_Psel, S=>add_O);
        
        -- this process routes the correct sub-operands of reg_A and reg_B to the STAR multiplier
        process(A, B, CTRL)
        begin
            if CTRL(6) = '0' then -- MUL32
                if CTRL(8) = '0' then -- AL (stage 00 o 01)
                    mul_A <= A(15 downto  0);
                else -- AH (stage 10 o 11)
                    mul_A <= A(31 downto 16);
                end if;
                if CTRL(7) = '0' then -- BL (stage 00 o 10)
                    mul_B <= B(15 downto  0);
                else -- BH (stage 01 o 11)
                    mul_B <= B(31 downto 16);
                end if;
            else -- STAR
                if CTRL(7) = '0' then -- AL * BL (stage 00)
                    mul_A <= A(15 downto  0);
                    mul_B <= B(15 downto  0);
                else -- AH * BH (stage 01)
                    mul_A <= A(31 downto 16);
                    mul_B <= B(31 downto 16);
                end if;
            end if;
        end process;

        -- this process generates mul_ctrl starting from the operation to be performed
        process(CTRL)
        begin
            if CTRL(6) = '0' then -- MUL32
                if CTRL(8 downto 7) = "00" then -- stage 00
                    mul_ctrl <= "000";
                elsif CTRL(8 downto 7) = "01" then -- stage 01
                    mul_ctrl(0) <= '0';
                    mul_ctrl(1) <= CTRL(1);
                    mul_ctrl(2) <= '0';
                elsif CTRL(8 downto 7) = "10" then -- stage 10
                    mul_ctrl(0) <= CTRL(0);
                    mul_ctrl(1) <= '0';
                    mul_ctrl(2) <= '0';
                else -- stage 11
                    mul_ctrl(0) <= CTRL(0);
                    mul_ctrl(1) <= CTRL(1);
                    mul_ctrl(2) <= '0';
                end if;
            else -- STAR
                if CTRL(4) = '0' then -- 16b
                    mul_ctrl <= "011";
                else
                    mul_ctrl(0) <= CTRL(3);
                    mul_ctrl(1) <= CTRL(5);
                    mul_ctrl(2) <= '1';
                end if;
            end if;
        end process;

        --this process manages routing to the "extended_adder" (input A) -- connected to the multiplier, surely CP
        process(CTRL, mul_O)
        begin
            add_A(31 downto 0) <= mul_O;

            --default
            add_A(51 downto 32) <= (others => '0');

            if CTRL(6) = '0' then -- MUL32
                if ((CTRL(0) = '1') and (CTRL(8) = '1')) or ((CTRL(0) = '1') and (CTRL(1) = '1')) then
                    add_A(51 downto 32) <= (others => mul_O(31));
                end if;
            else -- STAR
                if CTRL(4) = '0' then -- 16b
                    add_A(51 downto 32) <= (others => mul_O(31));
                else
                    if CTRL(5) = '0' then -- 8b
                        add_A(36 downto 32) <= (others => mul_O(31));
                        add_A(51 downto 37) <= (others => mul_O(15));
                    else -- 4b
                        add_A(36 downto 32) <= (others => mul_O(31));
                        add_A(41 downto 37) <= (others => mul_O(23));
                        add_A(46 downto 42) <= (others => mul_O(15));
                        add_A(51 downto 47) <= (others => mul_O( 7));
                    end if;
                end if;
            end if;
        end process;

        --this process manages routing to the "extended_adder" (input B) -- connected to the register, surely non-CP
        process(CTRL, MAC_REGi, ALU_REGi)
        begin
            add_B(51 downto 0) <= MAC_REGi(51 downto 0); -- default

            if CTRL(6) = '0' then -- MUL32
                add_B(33 downto 0) <= (others => '0'); -- stage 00 (default)
                if CTRL(8 downto 7) = "01" then -- stage 01
                    add_B(15 downto  0) <= ALU_REGi(31 downto 16);
                elsif CTRL(8 downto 7) = "10" then -- stage 10
                    if CTRL(4) = '1' then -- MULH
                        add_B(33 downto  0) <= ALU_REGi(33 downto 0);
                    else -- MULL
                        add_B(17 downto  0) <= ALU_REGi(33 downto 16);
                    end if;
                    elsif CTRL(8 downto 7) = "11" then -- stage 11
                    add_B(17 downto 0) <= ALU_REGi(33 downto 16);
                    add_B(33 downto 18) <= (others => ALU_REGi(33));
                end if;
            else -- STAR
                if CTRL(3) = '1' then -- SA
                    if CTRL(7) = '1' then -- stage "01"
                        add_B(51 downto 0) <= MAC_REGi(103 downto 52);
                    end if;
                end if;
            end if;
        end process;

        -- this process generates add_ctrl starting from the operation to be performed
        process(CTRL)
        begin
            -- default (ok for MUL32, ST16, ST8 e ST4)
            add_Psel <= "111"; -- no masking process
            add_ctrl_mux0 <= '0';
            add_ctrl_mux1 <= '0';
            add_ctrl_mux2 <= '0';
            if CTRL(6) = '1' then -- STAR
                if CTRL(3) = '1' then -- SA
                    if CTRL(4) = '1' then -- not 16b
                        if CTRL(5) = '0' then -- 8b
                            add_Psel <= "101";
                            add_ctrl_mux1 <= '1';
                        else -- 4b
                            add_Psel <= "000";
                            add_ctrl_mux0 <= '1';
                            add_ctrl_mux1 <= '1';
                            add_ctrl_mux2 <= '1';
                        end if;
                    end if;
                end if;
            end if;
        end process;

        --this process manages routing to the 104b register
        process(CTRL, MAC_REGi, add_O)
        begin
            -- default -> keep the value
            MAC_REGo(103 downto 0) <= MAC_REGi(103 downto 0);

            if CTRL(6) = '1' then -- STAR
                if CTRL(4) = '0' then -- 16b
                    if CTRL(3) = '0' then -- ST
                        if CTRL(2) = '0' then -- non ST_H
                            MAC_REGo(51 downto 0) <= add_O(51 downto 0);
                        end if;
                    else -- SA
                        if (CTRL(2) = '0') and (CTRL(0) = '0') then -- SA_L0
                            if CTRL(7) = '0' then -- stage "00"
                                MAC_REGo( 51 downto  0) <= add_O(51 downto 0);
                            else -- stage "11"
                                MAC_REGo(103 downto 52) <= add_O(51 downto 0);
                            end if;
                        end if;
                    end if;
                else
                    if CTRL(5) = '0' then -- 8b
                        if CTRL(3) = '0' then -- ST
                            MAC_REGo(51 downto 0) <= add_O(51 downto 0);
                        else -- SA
                            if CTRL(1 downto 0) = "00" then -- SA_L0
                                if CTRL(7) = '0' then -- stage "00"
                                    MAC_REGo( 51 downto  0) <= add_O(51 downto 0);
                                else -- stage "11"
                                    MAC_REGo(103 downto 52) <= add_O(51 downto 0);
                                end if;
                            end if;
                        end if;
                    else -- 4b
                        if CTRL(3) = '0' then -- ST
                            MAC_REGo(51 downto 0) <= add_O(51 downto 0);
                        else -- SA
                            if CTRL(2 downto 0) = "000" then -- SA_L0
                                if CTRL(7) = '0' then -- stage "00"
                                    MAC_REGo( 51 downto  0) <= add_O(51 downto 0);
                                else -- stage "11"
                                    MAC_REGo(103 downto 52) <= add_O(51 downto 0);
                                end if;
                            end if;
                        end if;
                    end if;
                end if;
            end if;
        end process;

        --this process manages the update to the ALU_REG --may be another CP
        process(CTRL, ALU_REGi, add_O)
        begin
            -- default -> update whole content from add_O
            ALU_REGo(33 downto 0) <= add_O(33 downto 0); -- stage 00 (default)
            if CTRL(8 downto 7) = "01" then -- stage 01
                if CTRL(4) = '0' then -- MULL
                    ALU_REGo(33 downto 16) <= add_O(17 downto 0);
                    ALU_REGo(15 downto  0) <= ALU_REGi(15 downto 0);
                end if;
                                            -- stage 10 (default)
                                            -- stage 11 (default)
            end if;
        end process;
        
        --these two processes manage routing from register and extended adder to output (they try to place MUXs with connection coming from the extended adder in the last stage)
        process(CTRL, MAC_REGi, ALU_REGi) -- this is not in the CP
        begin

            from_reg <= MAC_REGi(31 downto 0);

            if CTRL(6) = '0' then -- MUL32
                from_reg <= ALU_REGi(31 downto 0);
            else -- STAR
                if CTRL(4) = '0' then -- 16b
                    if CTRL(3) = '0' then -- ST
                        if CTRL(2) = '1' then -- ST_H
                            from_reg(15 downto  0) <= MAC_REGi(47 downto 32);
                            from_reg(31 downto 16) <= (others => MAC_REGi(47));
                        end if;
                    else -- SA
                        if CTRL(0) = '0' then -- SA_0
                            if CTRL(2) = '1' then -- SA_0H
                                from_reg( 4 downto 0) <= MAC_REGi(36 downto 32);
                                from_reg(31 downto 5) <= (others => MAC_REGi(36));
                            end if;
                        else -- SA_1
                            if CTRL(2) = '1' then -- SA_1H
                                from_reg( 4 downto 0) <= MAC_REGi(88 downto 84);
                                from_reg(31 downto 5) <= (others => MAC_REGi(88));
                            else -- SA_1L
                                from_reg(31 downto 0) <= MAC_REGi(83 downto 52);
                            end if;
                        end if;
                    end if;
                else
                    if CTRL(5) = '0' then -- 8b
                        if CTRL(3) = '1' then -- SA
                            if CTRL(1 downto 0) = "00" then -- SA_L0
                                from_reg(20 downto 16) <= MAC_REGi(46 downto 42);
                                from_reg(31 downto 21) <= (others => MAC_REGi(46));
                            elsif CTRL(1 downto 0) = "01" then -- SA_L1
                                from_reg(15 downto  0) <= MAC_REGi(31 downto 16);
                                from_reg(20 downto 16) <= MAC_REGi(36 downto 32);
                                from_reg(31 downto 21) <= (others => MAC_REGi(36));
                            elsif CTRL(1 downto 0) = "10" then -- SA_L2
                                from_reg(15 downto  0) <= MAC_REGi(67 downto 52);
                                from_reg(20 downto 16) <= MAC_REGi(98 downto 94);
                                from_reg(31 downto 21) <= (others => MAC_REGi(98));
                            else -- SA_L3
                                from_reg(15 downto  0) <= MAC_REGi(83 downto 68);
                                from_reg(20 downto 16) <= MAC_REGi(88 downto 84);
                                from_reg(31 downto 21) <= (others => MAC_REGi(88));
                            end if;
                        end if;
                    else -- 4b
                        if CTRL(3) = '1' then -- SA
                            if CTRL(2 downto 0) = "000" then -- SA_L0
                                from_reg(12 downto  8) <= MAC_REGi(51 downto 47);
                                from_reg(31 downto 12) <= (others => MAC_REGi(51));
                            elsif CTRL(2 downto 0) = "001" then -- SA_L1
                                from_reg( 7 downto  0) <= MAC_REGi(15 downto 8);
                                from_reg(12 downto  8) <= MAC_REGi(46 downto 42);
                                from_reg(31 downto 12) <= (others => MAC_REGi(46));
                            elsif CTRL(2 downto 0) = "010" then -- SA_L2
                                from_reg( 7 downto  0) <= MAC_REGi(23 downto 16);
                                from_reg(12 downto  8) <= MAC_REGi(41 downto 37);
                                from_reg(31 downto 12) <= (others => MAC_REGi(41));
                            elsif CTRL(2 downto 0) = "011" then -- SA_L3
                                from_reg( 7 downto  0) <= MAC_REGi(31 downto 24);
                                from_reg(12 downto  8) <= MAC_REGi(36 downto 32);
                                from_reg(31 downto 12) <= (others => MAC_REGi(36));
                            elsif CTRL(2 downto 0) = "100" then -- SA_L4
                                from_reg( 7 downto  0) <= MAC_REGi(59 downto 52);
                                from_reg(12 downto  8) <= MAC_REGi(103 downto 99);
                                from_reg(31 downto 12) <= (others => MAC_REGi(103));
                            elsif CTRL(2 downto 0) = "101" then -- SA_L5
                                from_reg( 7 downto  0) <= MAC_REGi(67 downto 60);
                                from_reg(12 downto  8) <= MAC_REGi(98 downto 94);
                                from_reg(31 downto 12) <= (others => MAC_REGi(98));
                            elsif CTRL(2 downto 0) = "110" then -- SA_L6
                                from_reg( 7 downto  0) <= MAC_REGi(75 downto 68);
                                from_reg(12 downto  8) <= MAC_REGi(93 downto 89);
                                from_reg(31 downto 12) <= (others => MAC_REGi(93));
                            else -- SA_L7
                                from_reg( 7 downto  0) <= MAC_REGi(83 downto 76);
                                from_reg(12 downto  8) <= MAC_REGi(88 downto 84);
                                from_reg(31 downto 12) <= (others => MAC_REGi(88));
                            end if;
                        end if;
                    end if;
                end if;
            end if;
        
        end process;

        -- this process select output signal from add_o or from_reg signals depending on the operation
        process(CTRL, from_reg, add_O) -- this is in the CP for sure
        begin
            RESo <= add_O(31 downto 0);

            if CTRL(6) = '0' then -- MUL32
                if CTRL(4) = '0' then -- MUL
                    RESo(15 downto  0) <= from_reg(15 downto 0);
                    RESo(31 downto 16) <= add_O(15 downto 0);
                end if;
            end if;

            if CTRL(6) = '1' then -- STAR
                if CTRL(4) = '0' then -- 16b
                    if CTRL(3) = '0' then -- ST
                        if CTRL(2) = '1' then -- ST_H
                            RESo <= from_reg;
                        end if;
                    else -- SA
                        RESo <= from_reg;
                    end if;
                else
                    if CTRL(3) = '1' then -- SA
                        RESo <= from_reg;
                    end if;
                end if;
            end if;

        end process;

end architecture;
