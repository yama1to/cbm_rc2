function delete(){
    DIR="./trashfigure/";

    echo -n "$*[Y/n]:"

    cd $DIR;

    read ANS

    case $ANS in
    "" | [Yy]* )
        # ここに「Yes」の時の処理を書く
        echo "Yes"
        rm -r generate_datasets/ipc_dir/*
        ;;
    * )
        # ここに「No」の時の処理を書く
        echo "No"
        ;;
    esac

}

delete "全保存画像を削除しますか？";
